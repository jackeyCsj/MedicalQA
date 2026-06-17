"""调试单条问答链路的中间输出。

用法示例：
    python debug_pipeline.py "我牙齿上长黑斑怎么回事"
    python debug_pipeline.py "高血压吃什么药好"
    python debug_pipeline.py "感冒了怎么办"
    python debug_pipeline.py "我脚崴了怎么办"
"""

from __future__ import annotations

import argparse
import logging
import time

import py2neo

from config import settings
from keyword_extractor import extract_keywords
from kg_client import KGClient
from prompt_builder import build_prompt
from retriever import KEYWORD_SCORE_WEIGHT, QUERY_SCORE_WEIGHT, Retriever
from vector_index import VectorIndex
from webui import llm_generate

# 让 retriever 的 debug 日志在 debug_pipeline 中可见
logging.getLogger("retriever").setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("  %(message)s"))
logging.getLogger("retriever").addHandler(_handler)


def sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="打印医疗问答 RAG 链路的中间输出（LightRAG 范式）")
    parser.add_argument("query", help="用户问题")
    parser.add_argument("--model", default="", help="LLM 模型名，为空时使用配置默认值")
    parser.add_argument("--top-k", type=int, default=10, help="向量检索返回的最大实体数")
    parser.add_argument("--threshold", type=float, default=0.3, help="向量相似度阈值")
    parser.add_argument("--final-top-k", type=int, default=5, help="完整问题重排后保留的最终实体数（3~10）")
    args = parser.parse_args()
    final_top_k = max(3, min(10, args.final_top_k))

    sep("用户问题")
    print(args.query)

    # ── Step 1: 关键词提取 ──
    sep("Step 1: LLM 关键词提取")
    t0 = time.time()
    keywords = extract_keywords(args.query, model=args.model)
    t1 = time.time()
    print(f"  耗时: {t1-t0:.2f}s")
    print(f"  local_keywords  (具体实体): {keywords.local_keywords}")
    print(f"  global_keywords (主题概念): {keywords.global_keywords}")

    # ── Step 2: 加载资源 ──
    sep("Step 2: 加载向量索引 + Neo4j")
    t0 = time.time()
    vi = VectorIndex.load()
    graph = py2neo.Graph(
        settings.NEO4J_URL,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        name=settings.NEO4J_DBNAME,
    )
    kg = KGClient(graph)
    retriever = Retriever(vi, kg)
    t1 = time.time()
    print(f"  耗时: {t1-t0:.2f}s")
    print(f"  向量索引: {vi.total} 条实体")

    # ── Step 3: Low-level 检索（向量匹配） ──
    sep("Step 3: Low-level 检索 — 向量匹配 local_keywords + 完整问题重排")
    t0 = time.time()
    raw_hits = []
    for kw in keywords.local_keywords:
        hits = vi.search(kw, top_k=args.top_k, score_threshold=args.threshold)
        raw_hits.extend(hits)
        print(f"\n  关键词「{kw}」→ 匹配到 {len(hits)} 个实体:")
        for h in hits:
            print(f"    {h.score:.4f}  [{h.entity_type}]  {h.name}")

    raw_hits.sort(key=lambda x: x.score, reverse=True)
    seen = set()
    unique_hits = []
    for hit in raw_hits:
        if hit.name in seen:
            continue
        seen.add(hit.name)
        unique_hits.append(hit)
    query_scores = vi.score_texts(args.query, [hit.name for hit in unique_hits])
    ranked_hits = []
    for hit, query_score in zip(unique_hits, query_scores):
        rank_score = KEYWORD_SCORE_WEIGHT * hit.score + QUERY_SCORE_WEIGHT * query_score
        ranked_hits.append((hit, query_score, rank_score))
    ranked_hits.sort(key=lambda item: (item[2], item[0].score), reverse=True)

    print(f"\n  完整问题重排（最终保留 top {final_top_k}）:")
    for index, (hit, query_score, rank_score) in enumerate(ranked_hits, 1):
        status = "保留" if index <= final_top_k else "候选"
        print(
            f"    {status}  rank={rank_score:.4f}  keyword={hit.score:.4f}  query={query_score:.4f}  "
            f"[{hit.entity_type}]  {hit.name}"
        )
    t1 = time.time()
    print(f"\n  Step3 候选实体: {len(unique_hits)} → {min(final_top_k, len(unique_hits))}")
    print(f"  向量匹配与重排耗时: {t1-t0:.2f}s")

    # ── Step 4: 图谱扩展 ──
    sep("Step 4: 双层检索（向量匹配 + 图谱一跳邻居扩展）")
    t0 = time.time()
    retrieval = retriever.retrieve(
        keywords,
        args.query,
        top_k=args.top_k,
        score_threshold=args.threshold,
        final_entity_top_k=final_top_k,
    )
    t1 = time.time()
    print(f"  检索耗时: {t1-t0:.2f}s")
    print(f"  最终返回 {len(retrieval.entities)} 个实体\n")

    for i, ent in enumerate(retrieval.entities, 1):
        query_score = f", 问题相关度: {ent.query_score:.4f}" if ent.query_score is not None else ""
        rank_score = f", 综合分: {ent.rank_score:.4f}" if ent.rank_score is not None else ""
        print(f"  [{i}] 【{ent.entity_type}】{ent.name}  (相似度: {ent.score:.4f}{query_score}{rank_score})")

        if ent.attributes:
            print(f"      属性:")
            for attr, value in ent.attributes.items():
                display = value[:80] + "..." if len(value) > 80 else value
                print(f"        {attr}: {display}")

        if ent.neighbors:
            print(f"      关系:")
            for rel, items in ent.neighbors.items():
                display = "、".join(items[:10])
                if len(items) > 10:
                    display += f" ...共{len(items)}个"
                print(f"        {rel}: {display}")

        if not ent.attributes and not ent.neighbors:
            print(f"      (无属性和关系数据)")
        print()

    # ── Step 5: Prompt 组装 ──
    sep("Step 5: 组装最终 Prompt")
    prompt = build_prompt(args.query, retrieval)
    print(f"  Prompt 长度: {len(prompt)} 字符\n")
    print(prompt)

    # ── Step 6: LLM 生成回答 ──
    sep("Step 6: LLM 生成回答")
    t0 = time.time()
    answer = llm_generate(prompt, model=args.model)
    t1 = time.time()
    print(f"  耗时: {t1-t0:.2f}s")
    print(f"  模型: {args.model or settings.LLM_MODEL}")
    print(f"\n{answer}")

    # ── 总结 ──
    sep("数据流总结")
    print(f"  用户问题 → LLM提取关键词({len(keywords.local_keywords)}个local + {len(keywords.global_keywords)}个global)")
    print(f"  → 向量匹配实体 → 图谱扩展邻居 → 组装Prompt({len(prompt)}字符)")
    print(f"  → LLM生成回答({len(answer)}字符, 耗时{t1-t0:.2f}s)")


if __name__ == "__main__":
    main()




