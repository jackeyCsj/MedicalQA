"""医疗问答服务编排层。

本模块是多轮诊断对话的核心编排器，负责协调以下组件完成完整的问答流程：
- keyword_extractor: LLM 关键词提取（local + global）
- retriever: 双层检索器（FAISS 向量检索 + Neo4j 图谱扩展）
- prompt_builder: Prompt 组装（单轮/诊断/最终回答）
- webui: LLM 流式生成
- conversation_service: 数据库 CRUD
- diagnostic: 诊断状态机（路由决策、换题检测）

三种执行路径：
1. _stream_single_turn(): 单轮模式（CLARIFICATION_ENABLED=False），不走路由不存 DB
2. _stream_first_turn(): 多轮诊断的首轮，创建会话 + RAG 检索 + 追问
3. _stream_followup_turn(): 多轮诊断的后续轮次，加载状态 + 重评分 + 路由 + 追问/回答
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import AsyncIterator, Dict, List, Optional

import py2neo
from sqlalchemy.orm import Session

from config import settings
from kg_client import KGClient
from keyword_extractor import extract_keywords
from prompt_builder import build_diagnostic_prompt, build_final_prompt, build_prompt
from retriever import Retriever
from vector_index import VectorIndex
from webui import llm_chat_stream

from backend.services.conversation_service import (
    create_conversation,
    create_qa_turn,
    get_conversation,
    get_latest_qa_turn,
    get_conversation_messages,
    save_message,
    update_conversation_phase,
    update_diagnostic_state,
    update_qa_turn_assistant,
)
from backend.services.diagnostic import (
    DISCLAIMER,
    CandidateEntity,
    DiagnosticState,
    decide_action,
    detect_topic_change,
    generate_topic_id,
)

logger = logging.getLogger(__name__)


def _entities_to_candidate_dicts(entities) -> list[dict]:
    """将 RetrievalResult.entities 转为 CandidateEntity 的 dict 列表。

    候选实体池包含完整信息（评分、来源、属性、邻居），用于：
    1. 存入 qa_turns.candidate_pool_json（JSONB），供后续轮次加载
    2. 路由决策时转为 CandidateEntity 对象列表
    """
    candidates = []
    for ent in entities:
        candidates.append({
            "name": ent.name,
            "entity_type": ent.entity_type,
            "scores": {
                "score": ent.score,
                "query_score": ent.query_score or 0,
                "rank_score": ent.rank_score or ent.score,
            },
            "source": {
                "from_symptoms": [],
                "from_keywords": [],
                "matched_relations": [],
            },
            "attributes": ent.attributes,
            "neighbors": ent.neighbors,
        })
    return candidates


def _entities_to_prompt_dicts(entities) -> list[dict]:
    """将 RetrievalResult.entities 转为精简的 prompt 实体 dict 列表。

    只保留名称、类型和评分，不包含属性和邻居（prompt 中不需要这些细节）。
    存入 qa_turns.prompt_entities_json，用于记录实际送入 prompt 的实体子集。
    """
    return [
        {
            "name": ent.name,
            "entity_type": ent.entity_type,
            "scores": {
                "score": ent.score,
                "query_score": ent.query_score or 0,
                "rank_score": ent.rank_score or ent.score,
            },
        }
        for ent in entities
    ]


class MedicalQAService:
    """医疗问答服务的核心编排器。

    职责：
    1. 管理底层组件的生命周期（VectorIndex、KGClient、Retriever），懒加载 + 线程安全
    2. 提供 stream_answer() 入口，根据配置和 conversation_id 自动选择执行路径
    3. 协调关键词提取 → 检索 → Prompt 构建 → LLM 生成 → 状态更新的完整流程

    所有方法通过 SSE 事件流（yield tuple）向前端推送进度和结果。
    """

    def __init__(self) -> None:
        # 线程锁：保护懒加载过程，避免并发请求重复初始化
        self._lock = threading.Lock()
        self._vi: Optional[VectorIndex] = None   # FAISS 向量索引
        self._kg: Optional[KGClient] = None       # Neo4j 知识图谱客户端
        self._retriever: Optional[Retriever] = None  # 双层检索器

    def load(self) -> None:
        """懒加载底层组件（线程安全）。首次调用时初始化，后续调用跳过。"""
        with self._lock:
            if self._vi is None:
                self._vi = VectorIndex.load()
                self._vi._get_model()  # 预加载 embedding 模型
            if self._kg is None:
                graph = py2neo.Graph(
                    settings.NEO4J_URL,
                    user=settings.NEO4J_USER,
                    password=settings.NEO4J_PASSWORD,
                    name=settings.NEO4J_DBNAME,
                )
                self._kg = KGClient(graph)
            if self._retriever is None:
                self._retriever = Retriever(self._vi, self._kg)

    @property
    def retriever(self) -> Retriever:
        """获取检索器实例（自动触发懒加载）。"""
        if self._retriever is None:
            self.load()
        assert self._retriever is not None
        return self._retriever

    def get_models(self) -> List[str]:
        """返回当前 LLM 提供商支持的模型列表。"""
        if settings.LLM_PROVIDER == "ollama":
            return [settings.OLLAMA_QWEN_MODEL, settings.OLLAMA_LLAMA_MODEL]
        return [m.strip() for m in settings.LLM_MODELS.split(",") if m.strip()]

    async def stream_answer(
        self,
        message: str,
        model: str,
        final_entity_top_k: int = 5,
        conversation_id: Optional[int] = None,
        db: Optional[Session] = None,
        user_id: Optional[int] = None,
    ) -> AsyncIterator[tuple[str, str | Dict[str, object]]]:
        """问答入口：根据配置和参数自动选择执行路径。

        路由逻辑：
        1. CLARIFICATION_ENABLED=False 或无 db → 单轮模式（向后兼容）
        2. conversation_id=None → 多轮诊断的首轮（创建会话 + RAG + 追问）
        3. conversation_id 有值 → 多轮诊断的后续轮次（加载状态 + 重评分 + 路由）

        参数：
            message: 用户消息文本
            model: LLM 模型名称
            final_entity_top_k: 送入 prompt 的实体数量（3-10）
            conversation_id: 会话 ID（None 表示新会话）
            db: SQLAlchemy 数据库会话（多轮模式必须）
            user_id: 当前用户 ID（多轮模式必须）

        Yields:
            (event_type, data) 元组，event_type 为 "status"/"debug"/"token"/"action"/"done"/"error"
        """
        self.load()

        if not settings.CLARIFICATION_ENABLED or conversation_id is None and db is None:
            # 单轮模式（向后兼容）：不走路由，不存 DB
            async for event in self._stream_single_turn(message, model, final_entity_top_k):
                yield event
            return

        assert db is not None, "多轮诊断模式需要 db session"
        assert user_id is not None, "多轮诊断模式需要 user_id"

        if conversation_id is None:
            # 第 1 轮：创建会话 + 正常 RAG 检索 + 追问
            async for event in self._stream_first_turn(
                message, model, final_entity_top_k, db, user_id
            ):
                yield event
        else:
            # 第 2+ 轮：加载状态 + 重评分 + 路由决策 + 追问/回答
            async for event in self._stream_followup_turn(
                message, model, final_entity_top_k, db, user_id, conversation_id
            ):
                yield event

    # ============================================================
    #  单轮模式（向后兼容，不走路由不存 DB）
    # ============================================================

    async def _stream_single_turn(
        self, message: str, model: str, final_entity_top_k: int
    ) -> AsyncIterator[tuple[str, str | Dict[str, object]]]:
        """单轮问答流程：关键词提取 → 检索 → Prompt → LLM 回答。

        与多轮诊断的区别：
        - 不创建会话，不存 DB
        - 不区分候选池和 prompt 实体
        - 不走路由决策，直接让 LLM 回答
        - 适用于 CLARIFICATION_ENABLED=False 或旧版前端
        """
        # 步骤 1：LLM 提取关键词（local=具体实体，global=主题概念）
        yield "status", "正在提取关键词..."
        keywords = await asyncio.to_thread(extract_keywords, message, model)
        yield "debug", {
            "step": "keywords",
            "local_keywords": keywords.local_keywords,
            "global_keywords": keywords.global_keywords,
        }

        # 步骤 2：双层检索（FAISS 向量 + Neo4j 图谱扩展）
        yield "status", "正在检索知识图谱..."
        retrieval = await asyncio.to_thread(
            self.retriever.retrieve, keywords, message,
            final_entity_top_k=final_entity_top_k,
        )
        entities_dict = {
            ent.name: {
                "type": ent.entity_type,
                "score": ent.score,
                "query_score": ent.query_score,
                "rank_score": ent.rank_score,
                "attributes": ent.attributes,
                "neighbors": ent.neighbors,
            }
            for ent in retrieval.entities
        }
        yield "debug", {"step": "retrieval", "final_entity_top_k": final_entity_top_k, "entities": entities_dict}

        # 步骤 3：组装单轮 prompt（沿用原有模板）
        prompt = build_prompt(message, retrieval)
        yield "debug", {
            "step": "prompt",
            "keywords": {"local": keywords.local_keywords, "global": keywords.global_keywords},
            "final_entity_top_k": final_entity_top_k,
            "knowledge": _format_knowledge(retrieval),
            "prompt": prompt,
        }

        # 步骤 4：LLM 流式生成回答
        yield "status", "正在生成回答..."
        async for chunk in self._stream_llm(prompt, model):
            if chunk:
                yield "token", chunk

        # 步骤 5：返回完成事件
        yield "done", {
            "entities": entities_dict,
            "keywords": {"local": keywords.local_keywords, "global": keywords.global_keywords},
            "final_entity_top_k": final_entity_top_k,
            "knowledge": _format_knowledge(retrieval),
        }

    # ============================================================
    #  第 1 轮：创建会话 + 正常 RAG + 追问
    # ============================================================

    async def _stream_first_turn(
        self,
        message: str,
        model: str,
        final_entity_top_k: int,
        db: Session,
        user_id: int,
    ) -> AsyncIterator[tuple[str, str | Dict[str, object]]]:
        """多轮诊断的首轮流程。

        与后续轮次的区别：
        - 创建新会话和初始诊断状态
        - 使用标准 retrieve() 而非 retrieve_followup()
        - 候选池设为 20（比 prompt 展示的 top-K 更大），为后续轮次提供充足的候选
        - 第 1 轮总是追问（不走路由决策），因为首轮查询通常太宽泛

        流程：
        1. 创建会话 + 保存用户消息
        2. 关键词提取
        3. 双层检索（候选池 20，prompt 展示 top-K）
        4. 构建诊断 prompt（追问用）
        5. 保存 QA 轮次快照
        6. LLM 流式生成追问内容
        7. 保存助手消息 + 更新诊断状态（turn=2）
        """

        # 步骤 1：创建会话
        #   phase 初始为 "clarifying"（首轮就进入追问阶段）
        #   topic_id 用于标识同一诊断主题（换题检测用）
        state = DiagnosticState(
            turn=1,
            phase="clarifying",
            max_turns=settings.MAX_CLARIFICATION_TURNS,
            topic_id=generate_topic_id(),
        )
        title = message[:50] + ("..." if len(message) > 50 else "")
        conv = create_conversation(db, user_id, title, state.to_dict())

        # 步骤 2：保存用户消息到 messages 表
        user_msg = save_message(db, conv.id, "user", message)

        # 步骤 3：LLM 提取关键词
        yield "status", "正在提取关键词..."
        keywords = await asyncio.to_thread(extract_keywords, message, model)
        yield "debug", {
            "step": "keywords",
            "local_keywords": keywords.local_keywords,
            "global_keywords": keywords.global_keywords,
        }

        # 步骤 4：双层检索（候选池设为 20，比 prompt 展示数更大）
        #   max_expanded_diseases=20: 从症状展开更多疾病候选
        #   final_entity_top_k 至少为 8: 保证 prompt 中有足够的实体上下文
        yield "status", "正在检索知识图谱..."
        retrieval = await asyncio.to_thread(
            self.retriever.retrieve, keywords, message,
            final_entity_top_k=max(final_entity_top_k, 8),
            max_expanded_diseases=20,
        )

        # 步骤 5：区分候选池（完整）和 prompt 实体（精简）
        #   candidate_pool: 全部检索到的实体（含评分、属性、邻居），存入 DB 供后续轮次用
        #   prompt_entities: rank_score 前 final_entity_top_k 个，只送入 prompt
        candidate_pool = _entities_to_candidate_dicts(retrieval.entities)
        prompt_entities = retrieval.entities[:final_entity_top_k]
        prompt_retrieval = type(retrieval)(entities=prompt_entities, matched_keywords=retrieval.matched_keywords)

        entities_dict = {
            ent.name: {
                "type": ent.entity_type,
                "score": ent.score,
                "query_score": ent.query_score,
                "rank_score": ent.rank_score,
            }
            for ent in retrieval.entities
        }
        yield "debug", {"step": "retrieval", "final_entity_top_k": final_entity_top_k, "entities": entities_dict}

        # 步骤 6：构建追问 prompt（用精简的 prompt_entities，不用完整候选池）
        prompt = build_diagnostic_prompt(
            query=message,
            retrieval=prompt_retrieval,
            symptoms=[kw for kw in keywords.local_keywords],
            candidate_pool_dicts=candidate_pool,  # 候选池信息写入 prompt 供 LLM 参考
            conversation_history=[],  # 首轮无历史
        )
        yield "debug", {"step": "prompt", "prompt": prompt}

        # 步骤 7：保存 QA 轮次快照（RAG 系统数据，与用户消息分离）
        qa_turn = create_qa_turn(
            db, conv.id, user_msg.id,
            mode="initial",  # 标记为首轮 RAG
            turn_number=1,
            keywords_json={"local": keywords.local_keywords, "global": keywords.global_keywords},
            candidate_pool_json=candidate_pool,  # 完整候选池存入 DB
            prompt_entities_json=_entities_to_prompt_dicts(prompt_entities),  # 精简版
            memory_state_json=state.to_dict(),
        )

        # 步骤 8：LLM 流式生成追问内容
        yield "status", "正在生成回答..."
        full_response = ""
        async for chunk in self._stream_llm(prompt, model):
            if chunk:
                full_response += chunk
                yield "token", chunk

        # 步骤 9：保存助手消息 + 回填 qa_turn 的 assistant_message_id
        assistant_msg = save_message(db, conv.id, "assistant", full_response)
        update_qa_turn_assistant(db, qa_turn, assistant_msg.id)

        # 步骤 10：更新诊断状态
        #   symptoms: 从 local_keywords 中提取（首轮所有关键词都算作症状）
        #   all_entity_names: 记录所有出现过的实体名（换题检测用）
        #   turn: 设为 2（下一轮是第 2 轮）
        state.symptoms = list(keywords.local_keywords)
        state.all_entity_names = [ent.name for ent in retrieval.entities]
        state.turn = 2
        update_diagnostic_state(db, conv, state.to_dict())

        # 步骤 11：返回事件流
        yield "action", {"type": "ask"}  # 首轮总是追问
        yield "done", {
            "conversation_id": conv.id,  # 前端保存此 ID，后续轮次传回
            "entities": entities_dict,
            "keywords": {"local": keywords.local_keywords, "global": keywords.global_keywords},
            "diagnostic_state": state.to_dict(),
        }

    # ============================================================
    #  第 2+ 轮：加载状态 + 重评分 + 路由 + 追问/回答
    # ============================================================

    async def _stream_followup_turn(
        self,
        message: str,
        model: str,
        final_entity_top_k: int,
        db: Session,
        user_id: int,
        conversation_id: int,
    ) -> AsyncIterator[tuple[str, str | Dict[str, object]]]:
        """多轮诊断的后续轮次流程（第 2 轮及之后）。

        与首轮的核心区别：
        - 从 DB 加载上轮的候选实体池，用 retrieve_followup() 做证据文本重评分
        - 经过规则路由决策（decide_action），决定追问还是回答
        - 支持换题检测：用户中途换话题时重置诊断状态
        - 最终回答时追加医学免责声明

        流程：
        1. 加载会话状态 + 上轮候选池
        2. 关键词提取 + 后续轮次检索（证据文本重评分）
        3. 换题检测（如果触发则重置状态，用新关键词做全量检索）
        4. 合并新旧候选实体
        5. 规则路由决策 → "ask" 或 "answer"
        6. 构建对应 prompt（追问用 / 最终回答用）
        7. LLM 流式生成 + 追加免责声明（如果 answer）
        8. 保存所有数据 + 更新诊断状态
        """

        # 步骤 1：加载会话和诊断状态
        conv = get_conversation(db, conversation_id, user_id)
        if not conv:
            yield "error", {"message": "会话不存在或无权访问"}
            return

        state = DiagnosticState.from_dict(conv.diagnostic_state or {})

        # 步骤 2：保存用户消息
        user_msg = save_message(db, conv.id, "user", message)

        # 步骤 3：加载上轮候选池（从 qa_turns.candidate_pool_json）
        prev_turn = get_latest_qa_turn(db, conv.id)
        previous_candidates = prev_turn.candidate_pool_json if prev_turn else []

        # 步骤 4：LLM 提取本轮关键词
        yield "status", "正在提取关键词..."
        keywords = await asyncio.to_thread(extract_keywords, message, model)
        yield "debug", {
            "step": "keywords",
            "local_keywords": keywords.local_keywords,
            "global_keywords": keywords.global_keywords,
        }

        # 合并初始轮次和本轮的 global_keywords
        #   初始轮次可能包含"治疗方法"、"治疗药品"等意图，后续轮次的新消息
        #   可能不再提及这些意图（如用户描述症状时），但初始意图不应丢失
        prev_global = []
        if prev_turn and prev_turn.keywords_json:
            prev_global = prev_turn.keywords_json.get("global", [])
        merged_global = list(dict.fromkeys(prev_global + keywords.global_keywords))
        logger.info("[followup] 本轮 global_keywords=%s, 初始轮 global_keywords=%s, 合并后=%s",
                    keywords.global_keywords, prev_global, merged_global)

        # 步骤 5：后续轮次检索（证据文本重评分 + 补充新实体）
        #   retrieve_followup() 返回 (RetrievalResult, all_below_threshold)
        #   all_below_threshold=True 表示所有候选重评分 < 0.2（可能是换题）
        yield "status", "正在检索知识图谱..."
        retrieval, all_below = await asyncio.to_thread(
            self.retriever.retrieve_followup,
            keywords, message, previous_candidates,
            merged_global,  # 合并后的 global_keywords，保留初始轮的治疗意图
            top_k=10,
            final_entity_top_k=final_entity_top_k,
        )

        # 步骤 6：换题检测
        #   条件：重评分全低于阈值 AND 新关键词与已有实体无交集
        #   如果触发：重置诊断状态，用新关键词做全量检索（相当于新话题的首轮）
        if all_below and detect_topic_change(keywords.local_keywords, state, all_below):
            logger.info("[followup] 检测到换题，重置诊断状态")
            state = DiagnosticState(
                turn=1,
                phase="clarifying",
                max_turns=settings.MAX_CLARIFICATION_TURNS,
                topic_id=generate_topic_id(),
            )
            # 用新关键词做全量检索（不使用 retrieve_followup，因为旧候选池已无关）
            retrieval = await asyncio.to_thread(
                self.retriever.retrieve, keywords, message,
                final_entity_top_k=final_entity_top_k,
            )
            previous_candidates = []

        # 步骤 7：构建候选实体列表（合并新旧候选）
        #   本轮检索到的实体优先，上轮未被覆盖的实体保留
        candidate_pool = _entities_to_candidate_dicts(retrieval.entities)
        if previous_candidates:
            existing_names = {c["name"] for c in candidate_pool}
            for pc in previous_candidates:
                if pc["name"] not in existing_names:
                    candidate_pool.append(pc)

        entities_dict = {
            ent.name: {
                "type": ent.entity_type,
                "score": ent.score,
                "query_score": ent.query_score,
                "rank_score": ent.rank_score,
            }
            for ent in retrieval.entities
        }
        yield "debug", {"step": "retrieval", "final_entity_top_k": final_entity_top_k, "entities": entities_dict}

        # 步骤 8：规则路由决策（不走 LLM，纯规则判断）
        candidate_entities = [CandidateEntity.from_dict(c) for c in candidate_pool]
        action = decide_action(state, candidate_entities)
        logger.info("[followup] 路由决策: %s (turn=%d)", action, state.turn)

        # 步骤 9：构建 prompt（根据路由结果选择追问模板或最终回答模板）
        #   history: 从 DB 加载已有对话历史（排除刚保存的用户消息，因为它是当前轮的输入）
        history = [
            {"role": m.role, "content": m.content}
            for m in get_conversation_messages(db, conv.id)[:-1]
        ]

        if action == "ask":
            # 追问 prompt：让 LLM 根据候选疾病和已收集症状提出有针对性的问题
            prompt = build_diagnostic_prompt(
                query=message,
                retrieval=retrieval,
                symptoms=state.symptoms + [kw for kw in keywords.local_keywords if kw not in state.symptoms],
                candidate_pool_dicts=candidate_pool,
                conversation_history=history,
            )
        else:
            # 最终回答 prompt：让 LLM 给出诊断分析和用药参考（含安全约束）
            prompt = build_final_prompt(
                query=message,
                retrieval=retrieval,
                symptoms=state.symptoms + [kw for kw in keywords.local_keywords if kw not in state.symptoms],
                candidate_pool_dicts=candidate_pool,
                conversation_history=history,
            )
        yield "debug", {"step": "prompt", "action": action, "prompt": prompt}

        # 步骤 10：保存 QA 轮次快照
        qa_turn = create_qa_turn(
            db, conv.id, user_msg.id,
            mode="followup",  # 标记为后续轮次
            turn_number=state.turn,
            keywords_json={"local": keywords.local_keywords, "global": keywords.global_keywords},
            candidate_pool_json=candidate_pool,
            prompt_entities_json=_entities_to_prompt_dicts(retrieval.entities),
            memory_state_json=state.to_dict(),
        )

        # 步骤 11：LLM 流式生成
        yield "status", "正在生成回答..."
        full_response = ""
        async for chunk in self._stream_llm(prompt, model):
            if chunk:
                full_response += chunk
                yield "token", chunk

        # 步骤 12：最终回答时追加医学免责声明（硬性约束，不可跳过）
        if action == "answer":
            yield "token", DISCLAIMER
            full_response += DISCLAIMER

        # 步骤 13：保存助手消息 + 回填 qa_turn
        assistant_msg = save_message(db, conv.id, "assistant", full_response)
        update_qa_turn_assistant(db, qa_turn, assistant_msg.id)

        # 步骤 14：更新诊断状态
        #   累积新症状（去重）→ 扩展已知实体名列表 → 递增轮次 → 更新阶段
        new_symptoms = [kw for kw in keywords.local_keywords if kw not in state.symptoms]
        state.symptoms.extend(new_symptoms)
        for ent in retrieval.entities:
            if ent.name not in state.all_entity_names:
                state.all_entity_names.append(ent.name)
        state.turn += 1
        state.phase = "answered" if action == "answer" else "clarifying"
        update_diagnostic_state(db, conv, state.to_dict())

        # 步骤 15：返回事件流
        yield "action", {"type": action}
        yield "done", {
            "conversation_id": conv.id,
            "entities": entities_dict,
            "keywords": {"local": keywords.local_keywords, "global": keywords.global_keywords},
            "diagnostic_state": state.to_dict(),
        }

    # ============================================================
    #  LLM 流式生成（桥接到 asyncio）
    # ============================================================

    async def _stream_llm(self, prompt: str, model: str) -> AsyncIterator[str]:
        """将同步的 LLM 流式生成桥接到 asyncio 事件循环。

        llm_chat_stream() 是同步生成器（阻塞 I/O），而 FastAPI 的 SSE 端点需要 async。
        解决方案：在独立线程中运行同步生成器，通过 asyncio.Queue 将 chunk 传递给 async 调用方。

        线程安全：使用 loop.call_soon_threadsafe() 从工作线程向事件循环线程投递消息。
        """
        queue: asyncio.Queue[tuple[str, Optional[str]]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            """工作线程：运行同步 LLM 生成器，将 chunk 投递到队列。"""
            try:
                for chunk in llm_chat_stream([{"role": "user", "content": prompt}], model=model):
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", chunk))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        # 启动守护线程，不阻塞事件循环
        threading.Thread(target=worker, daemon=True).start()
        while True:
            kind, value = await queue.get()
            if kind == "token" and value is not None:
                yield value
                continue
            if kind == "error":
                raise RuntimeError(value or "LLM streaming failed")
            break


def _format_knowledge(retrieval) -> str:
    """将检索结果格式化为知识图谱信息的纯文本表示。

    用于 debug 事件和单轮模式的 prompt 构建。
    格式：[实体类型]实体名: 属性1：值1; 关系1：值1、值2
    """
    knowledge_parts: List[str] = []
    for ent in retrieval.entities:
        lines = []
        for attr, value in ent.attributes.items():
            lines.append(f"{attr}：{value}")
        for rel, items in ent.neighbors.items():
            lines.append(f"{rel}：{'、'.join(items)}")
        if lines:
            knowledge_parts.append(f"[{ent.entity_type}]{ent.name}: {'; '.join(lines)}")
    return "\n".join(knowledge_parts)
