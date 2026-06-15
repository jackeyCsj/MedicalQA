"""构建实体向量索引。

从 data/ent_aug/*.txt 读取所有实体名，用 text2vec-base-chinese 生成 embedding，
存入 FAISS 索引。

用法：
    python build_vector_index.py
"""

import logging

from logging_setup import setup_logging
from vector_index import build_index

setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("开始构建实体向量索引...")
    idx = build_index()
    logger.info("构建完成! 索引包含 %d 条实体", idx.total)

    # 快速验证
    test_queries = ["牙齿上长黑斑", "高血压吃什么药", "感冒发烧"]
    for q in test_queries:
        results = idx.search(q, top_k=5)
        logger.info("查询: %s", q)
        for r in results:
            logger.info("  %.4f  [%s]  %s", r.score, r.entity_type, r.name)


if __name__ == "__main__":
    main()
