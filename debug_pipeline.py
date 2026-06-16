"""调试单条问答链路的中间输出。

用法示例：
    python debug_pipeline.py "我牙齿上长黑斑怎么回事"
    python debug_pipeline.py "高血压吃什么药好"
    python debug_pipeline.py "感冒了怎么办"
    python debug_pipeline.py "我脚崴了怎么办"
"""

from __future__ import annotations

import argparse
import json
import time

import py2neo

from config import settings
from keyword_extractor import extract_keywords
from kg_client import KGClient
from prompt_builder import build_prompt
from retriever import Retriever, DISEASE_ATTRIBUTES, DISEASE_RELATIONS
from vector_index import VectorIndex


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
    args = parser.parse_args()

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
    sep("Step 3: Low-level 检索 — 向量匹配 local_keywords")
    t0 = time.time()
    for kw in keywords.local_keywords:
        hits = vi.search(kw, top_k=args.top_k, score_threshold=args.threshold)
        print(f"\n  关键词「{kw}」→ 匹配到 {len(hits)} 个实体:")
        for h in hits:
            print(f"    {h.score:.4f}  [{h.entity_type}]  {h.name}")
    t1 = time.time()
    print(f"\n  向量匹配耗时: {t1-t0:.2f}s")

    # ── Step 4: 图谱扩展 ──
    sep("Step 4: 双层检索（向量匹配 + 图谱一跳邻居扩展）")
    t0 = time.time()
    retrieval = retriever.retrieve(keywords, top_k=args.top_k, score_threshold=args.threshold)
    t1 = time.time()
    print(f"  检索耗时: {t1-t0:.2f}s")
    print(f"  最终返回 {len(retrieval.entities)} 个实体\n")

    for i, ent in enumerate(retrieval.entities, 1):
        print(f"  [{i}] 【{ent.entity_type}】{ent.name}  (相似度: {ent.score:.4f})")

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
    print(prompt)

    # ── 总结 ──
    sep("数据流总结")
    print(f"  用户问题 → LLM提取关键词({len(keywords.local_keywords)}个local + {len(keywords.global_keywords)}个global)")
    print(f"  → 向量匹配实体 → 图谱扩展邻居 → 组装Prompt({len(prompt)}字符)")
    print(f"  → 送入LLM生成回答")


if __name__ == "__main__":
    main()
