"""调试单条问答链路的中间输出。

用法示例：
    python debug_pipeline.py "我牙齿上长黑斑怎么回事"
    python debug_pipeline.py "高血压吃什么药好"
    python debug_pipeline.py "感冒了怎么办"
"""

from __future__ import annotations

import argparse
import json

import py2neo

from config import settings
from keyword_extractor import extract_keywords
from kg_client import KGClient
from prompt_builder import build_prompt
from retriever import Retriever
from vector_index import VectorIndex


def print_section(title: str, value) -> None:
    print(f"\n==== {title} ====")
    if isinstance(value, (dict, list)):
        print(json.dumps(value, ensure_ascii=False, indent=2))
    else:
        print(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="打印医疗问答 RAG 链路的中间输出（LightRAG 范式）")
    parser.add_argument("query", help="用户问题")
    parser.add_argument("--model", default="", help="LLM 模型名，为空时使用配置默认值")
    parser.add_argument("--top-k", type=int, default=10, help="向量检索返回的最大实体数")
    parser.add_argument("--threshold", type=float, default=0.3, help="向量相似度阈值")
    args = parser.parse_args()

    # 1. 关键词提取
    print_section("用户问题", args.query)
    keywords = extract_keywords(args.query, model=args.model)
    print_section("Local Keywords", keywords.local_keywords)
    print_section("Global Keywords", keywords.global_keywords)

    # 2. 加载向量索引和知识图谱
    vi = VectorIndex.load()
    graph = py2neo.Graph(
        settings.NEO4J_URL,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        name=settings.NEO4J_DBNAME,
    )
    kg = KGClient(graph)
    retriever = Retriever(vi, kg)

    # 3. 双层检索
    retrieval = retriever.retrieve(keywords, top_k=args.top_k, score_threshold=args.threshold)
    for ent in retrieval.entities:
        info = {"type": ent.entity_type, "score": round(ent.score, 4)}
        if ent.attributes:
            info["attributes"] = ent.attributes
        if ent.neighbors:
            info["neighbors"] = ent.neighbors
        print_section(f"实体: {ent.name}", info)

    # 4. Prompt 组装
    prompt = build_prompt(args.query, retrieval)
    print_section("最终 Prompt", prompt)


if __name__ == "__main__":
    main()
