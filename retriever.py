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
from retrieval_intents import (
    DEFAULT_GLOBAL_KEYWORDS,
    DISEASE_ATTRIBUTE_INTENT_MAP,
    DISEASE_RELATION_INTENT_MAP,
)
from vector_index import EntityResult, VectorIndex

logger = logging.getLogger(__name__)

KEYWORD_SCORE_WEIGHT = 0.6
QUERY_SCORE_WEIGHT = 0.4

# 兼容调试脚本和旧引用；实际补充范围由 global_keywords 控制。
DISEASE_ATTRIBUTES = list(dict.fromkeys(DISEASE_ATTRIBUTE_INTENT_MAP.values()))
DISEASE_RELATIONS = [
    relation
    for relations in DISEASE_RELATION_INTENT_MAP.values()
    for relation in relations
]
GLOBAL_KEYWORD_RELATION_MAP = DISEASE_RELATION_INTENT_MAP


@dataclass
class RetrievedEntity:
    """检索到的单个实体及其关联信息。"""
    name: str
    entity_type: str
    score: float
    query_score: Optional[float] = None
    rank_score: Optional[float] = None
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
        query: str = "",
        top_k: int = 10,
        score_threshold: float = 0.3,
        final_entity_top_k: int = 5,
        max_diseases_per_symptom: int = 3,
        max_expanded_diseases: int = 10,
    ) -> RetrievalResult:
        """执行双层检索。

        :param keywords: LLM 提取的关键词
        :param query: 用户原始问题，用于完整问题相关性重排
        :param top_k: 每个 local keyword 向量检索返回的候选实体数
        :param score_threshold: 向量相似度阈值
        :param final_entity_top_k: 重排后送入图谱扩展的最终实体数
        :param max_diseases_per_symptom: 每个症状实体最多展开的疾病数
        :param max_expanded_diseases: 从症状展开疾病实体的总上限
        :return: 检索结果
        """
        result = RetrievalResult(matched_keywords=keywords)
        effective_global_keywords = self._effective_global_keywords(
            keywords.global_keywords
        )
        logger.debug("[retrieve] effective_global_keywords: %s", effective_global_keywords)

        # --- Low-level: local keywords → 向量匹配实体 ---
        if keywords.local_keywords:
            low_entities = self._low_level_retrieve(
                keywords.local_keywords, top_k, score_threshold
            )
            low_entities = self._rerank_by_query_relevance(
                low_entities, query, final_entity_top_k
            )
            for ent in low_entities:
                self._enrich_entity(ent, effective_global_keywords)
            result.entities.extend(low_entities)

        # --- High-level: global keywords → 控制一跳扩展方向 ---
        if result.entities:
            self._high_level_augment(
                result,
                effective_global_keywords,
                max_diseases_per_symptom,
                max_expanded_diseases,
            )

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
        """Low-level 检索：向量匹配候选实体。"""
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
        entities: List[RetrievedEntity] = []
        for r in unique:
            ent = RetrievedEntity(
                name=r.name, entity_type=r.entity_type, score=r.score
            )
            entities.append(ent)

        return entities

    def _rerank_by_query_relevance(
        self,
        entities: List[RetrievedEntity],
        query: str,
        final_entity_top_k: int,
    ) -> List[RetrievedEntity]:
        """用完整用户问题对候选实体重排，并保留最终 top-k。"""
        final_entity_top_k = max(3, min(10, final_entity_top_k))
        if not entities or not query.strip():
            return entities[:final_entity_top_k]

        scores = self.vi.score_texts(query, [ent.name for ent in entities])
        for ent, score in zip(entities, scores):
            ent.query_score = score
            ent.rank_score = (
                KEYWORD_SCORE_WEIGHT * ent.score + QUERY_SCORE_WEIGHT * score
            )

        entities.sort(
            key=lambda ent: (
                ent.rank_score if ent.rank_score is not None else ent.score,
                ent.score,
            ),
            reverse=True,
        )
        kept = entities[:final_entity_top_k]
        for idx, ent in enumerate(entities, 1):
            status = "保留" if idx <= final_entity_top_k else "候选"
            logger.debug(
                "[rerank] %s #%d %s/%s keyword_score=%.4f query_score=%.4f rank_score=%.4f",
                status,
                idx,
                ent.entity_type,
                ent.name,
                ent.score,
                ent.query_score or 0.0,
                ent.rank_score or ent.score,
            )
        logger.debug(
            "[rerank] 完整问题重排: %d → %d",
            len(entities),
            len(kept),
        )
        return kept

    def _enrich_entity(self, ent: RetrievedEntity, global_keywords: List[str]) -> None:
        """按 global_keywords 为实体补充属性和邻居信息。"""
        if ent.entity_type == "疾病":
            for gkw in global_keywords:
                attr = DISEASE_ATTRIBUTE_INTENT_MAP.get(gkw)
                if attr:
                    value = self.kg.get_disease_attribute(ent.name, attr)
                    if value:
                        ent.attributes[attr] = value
                        logger.debug("[enrich] 【疾病】%s → 意图 %s → 属性 %s = %s", ent.name, gkw, attr, value[:60])

                for rel, target in DISEASE_RELATION_INTENT_MAP.get(gkw, []):
                    items = self.kg.get_related_entities(ent.name, rel, target)
                    if items:
                        ent.neighbors[rel] = items
                        logger.debug("[enrich] 【疾病】%s → 意图 %s → 关系 %s: %s", ent.name, gkw, rel, "、".join(items[:5]))

        elif ent.entity_type == "疾病症状":
            # 通过症状反查疾病
            diseases = self.kg.get_diseases_by_symptom(ent.name)
            if diseases:
                ent.neighbors["可能的疾病"] = diseases
                logger.debug("[enrich] 【疾病症状】%s → 可能的疾病: %s", ent.name, "、".join(diseases[:5]))
            else:
                logger.debug("[enrich] 【疾病症状】%s → 无关联疾病", ent.name)

        elif ent.entity_type == "药品":
            # 查询药品生产商
            producers = self.kg.get_drug_producers(ent.name)
            if producers:
                ent.neighbors["生产商"] = producers
                logger.debug("[enrich] 【药品】%s → 生产商: %s", ent.name, "、".join(producers[:5]))

    def _high_level_augment(
        self,
        result: RetrievalResult,
        global_keywords: List[str],
        max_diseases_per_symptom: int = 3,
        max_expanded_diseases: int = 10,
    ) -> None:
        """High-level 检索：根据 global keywords 控制图谱扩展方向。

        从症状实体的"可能的疾病"邻居中，展开尚未出现的疾病实体。
        新增疾病实体只补充 global_keywords 对应的属性和关系。
        """
        # 统计当前实体类型分布
        type_counts: Dict[str, int] = {}
        for ent in result.entities:
            type_counts[ent.entity_type] = type_counts.get(ent.entity_type, 0) + 1
        logger.debug("[high-level] 当前实体类型分布: %s", type_counts)
        logger.debug("[high-level] global_keywords: %s", global_keywords)

        # --- 从症状的"可能的疾病"邻居中展开疾病实体 ---
        # 按"可能的疾病"数量升序：具体症状（疾病少）优先于泛化症状（疾病多）
        symptom_ents = sorted(
            [e for e in result.entities if e.entity_type == "疾病症状"],
            key=lambda e: len(e.neighbors.get("可能的疾病", [])),
        )
        existing_names = {ent.name for ent in result.entities}
        added = 0
        for ent in symptom_ents:
            if added >= max_expanded_diseases:
                break
            count = 0
            for dis_name in ent.neighbors.get("可能的疾病", []):
                if count >= max_diseases_per_symptom or added >= max_expanded_diseases:
                    break
                if dis_name in existing_names:
                    continue
                disease_ent = RetrievedEntity(
                    name=dis_name,
                    entity_type="疾病",
                    score=ent.score,
                    query_score=ent.query_score,
                    rank_score=ent.rank_score,
                )
                self._enrich_entity(disease_ent, global_keywords)
                result.entities.append(disease_ent)
                existing_names.add(dis_name)
                added += 1
                count += 1
                logger.debug(
                    "[high-level] 从症状「%s」展开疾病「%s」→ 属性 %d 个，关系 %d 类",
                    ent.name,
                    dis_name,
                    len(disease_ent.attributes),
                    len(disease_ent.neighbors),
                )
        if added:
            logger.debug("[high-level] 共从症状展开 %d 个疾病实体（每症状最多 %d，总上限 %d）",
                         added, max_diseases_per_symptom, max_expanded_diseases)

    def _effective_global_keywords(self, global_keywords: List[str]) -> List[str]:
        """返回可执行的 global keywords；为空时使用最小默认上下文。"""
        supported = set(DISEASE_ATTRIBUTE_INTENT_MAP) | set(DISEASE_RELATION_INTENT_MAP)
        result: List[str] = []
        seen: set[str] = set()
        for kw in global_keywords:
            if kw not in supported or kw in seen:
                continue
            seen.add(kw)
            result.append(kw)
        return result or list(DEFAULT_GLOBAL_KEYWORDS)
