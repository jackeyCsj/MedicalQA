"""双层检索器：替代意图路由 + Cypher 精确查询。

参考 LightRAG 的 dual-level retrieval paradigm：
- Low-level：local keywords → 向量匹配实体 → 图谱一跳邻居扩展
- High-level：global keywords → 向量匹配主题/关系 → 聚合相关实体群

返回结构化的检索结果，供 prompt_builder 组装最终 prompt。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from keyword_extractor import Keywords
from kg_client import KGClient
from vector_index import EntityResult, VectorIndex

logger = logging.getLogger(__name__)

# 疾病实体的所有属性
DISEASE_ATTRIBUTES = [
    "疾病简介", "疾病病因", "预防措施", "治疗周期", "治愈概率", "疾病易感人群",
]

# 疾病实体的关系及其目标标签
DISEASE_RELATIONS = [
    ("疾病使用药品", "药品"),
    ("疾病宜吃食物", "食物"),
    ("疾病忌吃食物", "食物"),
    ("疾病所需检查", "检查项目"),
    ("疾病所属科目", "科目"),
    ("疾病的症状", "疾病症状"),
    ("治疗的方法", "治疗方法"),
    ("疾病并发疾病", "疾病"),
]

# global_keywords 到关系的映射
GLOBAL_KEYWORD_RELATION_MAP: Dict[str, List[tuple]] = {
    "治疗药品": [("疾病使用药品", "药品")],
    "治疗方法": [("治疗的方法", "治疗方法")],
    "宜吃食物": [("疾病宜吃食物", "食物")],
    "忌吃食物": [("疾病忌吃食物", "食物")],
    "检查项目": [("疾病所需检查", "检查项目")],
    "并发疾病": [("疾病并发疾病", "疾病")],
    "症状": [("疾病的症状", "疾病症状")],
    "所属科目": [("疾病所属科目", "科目")],
}


@dataclass
class RetrievedEntity:
    """检索到的单个实体及其关联信息。"""
    name: str
    entity_type: str
    score: float
    attributes: Dict[str, str] = field(default_factory=dict)
    neighbors: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """双层检索的完整结果。"""
    entities: List[RetrievedEntity] = field(default_factory=list)
    matched_keywords: Keywords = field(default_factory=Keywords)


class Retriever:
    """双层检索器。

    :param vector_index: 向量索引实例
    :param kg: 知识图谱客户端
    """

    def __init__(self, vector_index: VectorIndex, kg: KGClient) -> None:
        self.vi = vector_index
        self.kg = kg

    def retrieve(
        self,
        keywords: Keywords,
        top_k: int = 10,
        score_threshold: float = 0.3,
    ) -> RetrievalResult:
        """执行双层检索。

        :param keywords: LLM 提取的关键词
        :param top_k: 向量检索返回的最大实体数
        :param score_threshold: 向量相似度阈值
        :return: 检索结果
        """
        result = RetrievalResult(matched_keywords=keywords)

        # --- Low-level: local keywords → 向量匹配实体 + 图谱扩展 ---
        if keywords.local_keywords:
            low_entities = self._low_level_retrieve(
                keywords.local_keywords, top_k, score_threshold
            )
            result.entities.extend(low_entities)

        # --- High-level: global keywords → 主题检索 ---
        if keywords.global_keywords and result.entities:
            self._high_level_augment(result, keywords.global_keywords)

        # 去重（按实体名）
        seen: set[str] = set()
        deduped: List[RetrievedEntity] = []
        for ent in result.entities:
            if ent.name not in seen:
                seen.add(ent.name)
                deduped.append(ent)
        result.entities = deduped

        logger.info("检索完成: %d 个实体", len(result.entities))
        return result

    def _low_level_retrieve(
        self,
        local_keywords: List[str],
        top_k: int,
        score_threshold: float,
    ) -> List[RetrievedEntity]:
        """Low-level 检索：向量匹配 + 一跳邻居扩展。"""
        # 1. 向量匹配
        all_results: List[EntityResult] = []
        for kw in local_keywords:
            hits = self.vi.search(kw, top_k=top_k, score_threshold=score_threshold)
            all_results.extend(hits)

        # 按分数排序去重
        all_results.sort(key=lambda x: x.score, reverse=True)
        seen: set[str] = set()
        unique: List[EntityResult] = []
        for r in all_results:
            if r.name not in seen:
                seen.add(r.name)
                unique.append(r)
        unique = unique[:top_k]

        # 2. 图谱扩展：查询属性和邻居
        entities: List[RetrievedEntity] = []
        for r in unique:
            ent = RetrievedEntity(
                name=r.name, entity_type=r.entity_type, score=r.score
            )
            self._enrich_entity(ent)
            entities.append(ent)

        return entities

    def _enrich_entity(self, ent: RetrievedEntity) -> None:
        """为实体补充属性和邻居信息。"""
        if ent.entity_type == "疾病":
            # 查询所有属性
            for attr in DISEASE_ATTRIBUTES:
                value = self.kg.get_disease_attribute(ent.name, attr)
                if value:
                    ent.attributes[attr] = value
            # 查询所有关系邻居
            for rel, target in DISEASE_RELATIONS:
                items = self.kg.get_related_entities(ent.name, rel, target)
                if items:
                    ent.neighbors[rel] = items

        elif ent.entity_type == "疾病症状":
            # 通过症状反查疾病
            diseases = self.kg.get_diseases_by_symptom(ent.name)
            if diseases:
                ent.neighbors["可能的疾病"] = diseases

        elif ent.entity_type == "药品":
            # 查询药品生产商
            producers = self.kg.get_drug_producers(ent.name)
            if producers:
                ent.neighbors["生产商"] = producers

    def _high_level_augment(
        self,
        result: RetrievalResult,
        global_keywords: List[str],
    ) -> None:
        """High-level 检索：根据 global keywords 补充关系信息。

        对已检索到的疾病实体，根据 global keywords 选择性地补充相关关系。
        """
        for gkw in global_keywords:
            relations = GLOBAL_KEYWORD_RELATION_MAP.get(gkw, [])
            for rel, target in relations:
                for ent in result.entities:
                    if ent.entity_type == "疾病" and rel not in ent.neighbors:
                        items = self.kg.get_related_entities(ent.name, rel, target)
                        if items:
                            ent.neighbors[rel] = items
