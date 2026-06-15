"""向量索引层：基于 FAISS 的实体语义检索。

为知识图谱中所有实体名称建立向量索引，支持语义匹配检索。
替代原 NER 的 TF-IDF 对齐 + Aho-Corasick 规则匹配。

使用方式：
    # 构建索引（一次性）
    python build_vector_index.py

    # 检索
    from vector_index import VectorIndex
    idx = VectorIndex.load()
    results = idx.search("牙齿上长黑斑", top_k=10)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# 索引文件默认存放路径
_INDEX_DIR = os.path.join("tmp_data", "vector_index")
_INDEX_FILE = os.path.join(_INDEX_DIR, "entities.faiss")
_META_FILE = os.path.join(_INDEX_DIR, "entities_meta.npy")
_EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"


@dataclass
class EntityResult:
    """向量检索返回的单条实体结果。"""
    name: str
    entity_type: str
    score: float


class VectorIndex:
    """FAISS 向量索引封装。

    通过 :meth:`load` 加载预构建的索引，通过 :meth:`search` 做语义检索。
    """

    def __init__(self, index: faiss.Index, names: np.ndarray, types: np.ndarray) -> None:
        self._index = index
        self._names = names   # shape (N,) dtype str
        self._types = types   # shape (N,) dtype str

    @classmethod
    def load(cls, index_dir: str = _INDEX_DIR) -> VectorIndex:
        """从磁盘加载预构建的索引。"""
        index_path = os.path.join(index_dir, "entities.faiss")
        meta_path = os.path.join(index_dir, "entities_meta.npy")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"向量索引文件不存在，请先运行 python build_vector_index.py\n"
                f"期望路径: {index_path}"
            )
        logger.info("加载向量索引: %s", index_path)
        index = faiss.read_index(index_path)
        meta = np.load(meta_path, allow_pickle=True)
        names = meta["names"]
        types = meta["types"]
        logger.info("索引加载完成: %d 条实体", len(names))
        return cls(index, names, types)

    def search(self, query: str, top_k: int = 20, score_threshold: float = 0.0) -> List[EntityResult]:
        """语义检索：返回与 query 最相似的 top_k 个实体。

        :param query: 查询文本
        :param top_k: 返回数量
        :param score_threshold: 最低相似度阈值（余弦相似度，0~1）
        :return: 按相似度降序排列的实体结果列表
        """
        embedding = self._encode_query(query)
        scores, indices = self._index.search(embedding, top_k)
        results: List[EntityResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < score_threshold:
                continue
            results.append(EntityResult(
                name=str(self._names[idx]),
                entity_type=str(self._types[idx]),
                score=float(score),
            ))
        return results

    def search_batch(self, queries: Sequence[str], top_k: int = 20) -> List[List[EntityResult]]:
        """批量语义检索。"""
        embeddings = self._encode_batch(list(queries))
        scores, indices = self._index.search(embeddings, top_k)
        batch_results: List[List[EntityResult]] = []
        for q_scores, q_indices in zip(scores, indices):
            results: List[EntityResult] = []
            for score, idx in zip(q_scores, q_indices):
                if idx < 0:
                    continue
                results.append(EntityResult(
                    name=str(self._names[idx]),
                    entity_type=str(self._types[idx]),
                    score=float(score),
                ))
            batch_results.append(results)
        return batch_results

    @property
    def total(self) -> int:
        return self._index.ntotal

    @staticmethod
    def _encode_query(query: str) -> np.ndarray:
        """单条查询编码为归一化向量。"""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(_EMBEDDING_MODEL)
        vec = model.encode([query], normalize_embeddings=True)
        return np.ascontiguousarray(vec, dtype="float32")

    @staticmethod
    def _encode_batch(texts: List[str]) -> np.ndarray:
        """批量编码为归一化向量。"""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(_EMBEDDING_MODEL)
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
        return np.ascontiguousarray(vecs, dtype="float32")


def build_index(
    entity_dir: str = os.path.join("data", "ent_aug"),
    output_dir: str = _INDEX_DIR,
    model_name: str = _EMBEDDING_MODEL,
) -> VectorIndex:
    """从 data/ent_aug/*.txt 读取实体名，构建 FAISS 索引并保存到磁盘。

    :param entity_dir: 实体文件目录
    :param output_dir: 索引输出目录
    :param model_name: sentence-transformers 模型名
    :return: 构建好的 VectorIndex 实例
    """
    from sentence_transformers import SentenceTransformer

    # 1. 读取所有实体
    names: List[str] = []
    types: List[str] = []
    for fname in sorted(os.listdir(entity_dir)):
        if not fname.endswith(".txt"):
            continue
        entity_type = fname.removesuffix(".txt")
        fpath = os.path.join(entity_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    names.append(name)
                    types.append(entity_type)
    logger.info("读取实体总数: %d", len(names))

    # 2. 批量编码
    logger.info("加载 embedding 模型: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("开始编码 %d 个实体...", len(names))
    embeddings = model.encode(names, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    embeddings = np.ascontiguousarray(embeddings, dtype="float32")
    logger.info("编码完成, shape=%s", embeddings.shape)

    # 3. 构建 FAISS 索引（内积索引 = 归一化向量的余弦相似度）
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS 索引构建完成, 总条目=%d", index.ntotal)

    # 4. 保存
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "entities.faiss"))
    np.savez(
        os.path.join(output_dir, "entities_meta.npy"),
        names=np.array(names, dtype=object),
        types=np.array(types, dtype=object),
    )
    logger.info("索引已保存到: %s", output_dir)
    return VectorIndex(index, np.array(names, dtype=object), np.array(types, dtype=object))
