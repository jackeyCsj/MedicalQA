"""诊断状态机：数据结构、路由决策、换题检测。

本模块是多轮诊断对话的核心逻辑层，负责：
1. 定义候选实体和诊断状态的数据结构（CandidateEntity、DiagnosticState）
2. 基于规则的路由决策（decide_action）：判断当前轮次应该继续追问还是给出回答
3. 换题检测（detect_topic_change）：判断用户是否在对话中途切换了话题

典型调用链：
    qa_service._stream_followup_turn() → decide_action() → "ask" | "answer"
    qa_service._stream_followup_turn() → detect_topic_change() → True | False
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# 症状类实体类型集合。从 local_keywords 中筛选出属于该类型的实体作为"症状"
SYMPTOM_ENTITY_TYPES = {"疾病症状"}

# 医学免责声明，每次最终回答后强制追加
DISCLAIMER = (
    "\n\n---\n"
    "**免责声明：以上信息仅供参考，基于知识图谱中的公开医疗数据整理，"
    "不构成医疗诊断或处方建议。请务必在专业医师的指导下就医，"
    "切勿自行用药。**"
)


@dataclass
class CandidateEntity:
    """候选实体，包含完整的证据信息（评分、来源、KG 属性和关系）。

    每个候选实体代表一个从知识图谱中检索到的可能匹配项（疾病、症状、药品等）。
    在多轮诊断中，候选实体池会被持久化到 qa_turns.candidate_pool_json，
    后续轮次通过 retrieve_followup() 对池内实体做证据文本重评分。

    字段说明：
        name: 实体名称（如"感冒"、"发烧"）
        entity_type: 实体类型（"疾病"、"疾病症状"、"药品"等）
        scores: 评分字典，包含三个维度：
            - score: 原始向量相似度分数（FAISS 内积）
            - query_score: 完整用户问题与实体名的语义相似度
            - rank_score: 综合排名分数（加权组合以上两者 + 后续轮次的症状匹配加分）
        source: 来源追踪，记录该候选实体是如何被检索命中的：
            - from_symptoms: 命中的症状关键词列表
            - from_keywords: 命中的所有关键词列表
            - matched_relations: 匹配的图谱关系类型
        attributes: 知识图谱中的实体属性（如疾病简介、病因、预防措施等）
        neighbors: 知识图谱中的一跳邻居关系（如关联症状、药品、治疗方法等）
    """
    name: str
    entity_type: str
    scores: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    attributes: dict = field(default_factory=dict)
    neighbors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为字典，用于存入 qa_turns.candidate_pool_json（JSONB 列）。"""
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "scores": self.scores,
            "source": self.source,
            "attributes": self.attributes,
            "neighbors": self.neighbors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateEntity":
        """从字典反序列化，用于从 qa_turns.candidate_pool_json 加载历史候选实体。"""
        return cls(
            name=data["name"],
            entity_type=data["entity_type"],
            scores=data.get("scores", {}),
            source=data.get("source", {}),
            attributes=data.get("attributes", {}),
            neighbors=data.get("neighbors", {}),
        )


@dataclass
class DiagnosticState:
    """诊断状态机的当前状态，以 JSONB 格式持久化在 conversations.diagnostic_state 中。

    每轮对话结束后更新此状态，下一轮从 DB 加载恢复。
    不存储候选疾病列表（那些存在 qa_turns.candidate_pool_json 中），
    只存轻量级的元数据，用于路由决策和换题检测。

    字段说明：
        turn: 当前轮次序号（从 1 开始），每轮 +1，达到 max_turns 时强制回答
        phase: 诊断阶段
            - "collecting": 初始状态，刚创建会话
            - "clarifying": 追问中，LLM 正在收集症状信息
            - "answered": 已给出最终回答
        max_turns: 最大追问轮次（由 config.py 的 MAX_CLARIFICATION_TURNS 控制）
        symptoms: 累积收集到的症状列表（从每轮 local_keywords 中提取并去重）
        all_entity_names: 所有轮次中出现过的实体名称（去重），用于换题检测
        topic_id: 当前诊断主题的唯一标识（首轮生成），用于关联同一诊断会话
    """
    turn: int = 1
    phase: str = "collecting"
    max_turns: int = 2
    symptoms: List[str] = field(default_factory=list)
    all_entity_names: List[str] = field(default_factory=list)
    topic_id: str = ""

    def to_dict(self) -> dict:
        """序列化为字典，用于存入 conversations.diagnostic_state（JSONB 列）。"""
        return {
            "turn": self.turn,
            "phase": self.phase,
            "max_turns": self.max_turns,
            "symptoms": self.symptoms,
            "all_entity_names": self.all_entity_names,
            "topic_id": self.topic_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DiagnosticState":
        """从字典反序列化，用于从 conversations.diagnostic_state 加载诊断状态。"""
        return cls(
            turn=data.get("turn", 1),
            phase=data.get("phase", "collecting"),
            max_turns=data.get("max_turns", 2),
            symptoms=data.get("symptoms", []),
            all_entity_names=data.get("all_entity_names", []),
            topic_id=data.get("topic_id", ""),
        )


def generate_topic_id() -> str:
    """生成 8 位十六进制的 topic_id，用于标识同一诊断会话的主题。"""
    return uuid.uuid4().hex[:8]


def decide_action(state: DiagnosticState, candidate_entities: List[CandidateEntity]) -> str:
    """规则优先的路由决策：返回 "ask"（继续追问）或 "answer"（给出最终回答）。

    决策逻辑按优先级依次判断，命中即返回，不走 LLM（节省调用成本，结果更稳定）：
        1. 超过最大追问轮次 → 强制回答（避免无限追问）
        2. 候选疾病只剩 1 个且评分 > 0.4 → 回答（疾病已明确）
        3. 候选疾病 2 个但分数断层明显（差 > 0.15 且 top > 0.6）→ 回答（可以区分）
        4. 已收集 3+ 个症状且 top 疾病评分 > 0.5 → 回答（信息充分）
        5. 以上都不满足 → 继续追问

    参数：
        state: 当前诊断状态（轮次、症状列表等）
        candidate_entities: 当前候选实体池（包含疾病和非疾病实体）
    """
    # 规则 1：超过最大轮次 → 强制回答，避免无限追问
    if state.turn >= state.max_turns:
        logger.info("[routing] turn %d >= max_turns %d → answer", state.turn, state.max_turns)
        return "answer"

    # 从候选池中筛选出疾病类型的实体
    diseases = [e for e in candidate_entities if e.entity_type == "疾病"]

    # 规则 2：候选疾病数量 == 1 且评分较高 → 疾病已经明确，直接回答
    if len(diseases) == 1 and diseases[0].scores.get("rank_score", 0) > 0.4:
        logger.info("[routing] 单一候选疾病 '%s' (score=%.2f) → answer",
                     diseases[0].name, diseases[0].scores.get("rank_score", 0))
        return "answer"

    # 规则 3：候选疾病 2 个，但分数断层明显 → top 疾病大概率是正确答案
    #   断层判断：top 和 second 的 rank_score 差 > 0.15，且 top 本身 > 0.6
    if len(diseases) == 2:
        sorted_scores = sorted(
            [d.scores.get("rank_score", 0) for d in diseases], reverse=True
        )
        if sorted_scores[0] - sorted_scores[1] > 0.15 and sorted_scores[0] > 0.6:
            logger.info("[routing] 分数断层: top=%.2f, second=%.2f → answer",
                         sorted_scores[0], sorted_scores[1])
            return "answer"

    # 规则 4：已收集 3+ 个症状，且 top 疾病评分 > 0.5 → 信息足够充分
    if len(state.symptoms) >= 3:
        top_diseases = sorted(diseases, key=lambda d: d.scores.get("rank_score", 0), reverse=True)
        if top_diseases and top_diseases[0].scores.get("rank_score", 0) > 0.5:
            logger.info("[routing] %d 个症状, top disease '%s' (score=%.2f) → answer",
                         len(state.symptoms), top_diseases[0].name,
                         top_diseases[0].scores.get("rank_score", 0))
            return "answer"

    # 规则都不命中 → 继续追问
    logger.info("[routing] 继续追问 (turn=%d, %d diseases, %d symptoms)",
                state.turn, len(diseases), len(state.symptoms))
    return "ask"


def detect_topic_change(
    new_local_keywords: List[str],
    state: DiagnosticState,
    rescored_all_below_threshold: bool,
) -> bool:
    """检测用户是否换了话题（如从"发烧"转到"牙疼"）。

    换题检测的两个必要条件同时满足才判定为换题：
        1. 新提取的关键词与已有实体名称没有任何交集（语义无关）
        2. 证据文本重评分后所有候选的 rank_score 都低于 0.2（旧候选不再匹配）

    仅满足条件 1 不算换题（可能是用户的补充描述）；
    仅满足条件 2 不算换题（可能只是关键词提取质量差）。

    参数：
        new_local_keywords: 本轮新提取的 local_keywords
        state: 当前诊断状态（包含 all_entity_names）
        rescored_all_below_threshold: retrieve_followup() 返回的标志，
            表示所有候选重评分是否都低于 0.2
    """
    # 首轮还没有历史实体，无法判断是否换题
    if not state.all_entity_names:
        return False

    # 计算新关键词与已有实体的交集
    new_kw_set = set(new_local_keywords)
    existing_set = set(state.all_entity_names)
    overlap = new_kw_set & existing_set

    # 两个条件同时满足 → 判定为换题
    if not overlap and rescored_all_below_threshold:
        logger.info("[topic_change] 新关键词 %s 与已有实体 %s 无交集，且重评分全低于阈值",
                     new_local_keywords, state.all_entity_names[:5])
        return True

    return False
