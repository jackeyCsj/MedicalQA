"""LightRAG 风格的 Prompt 组装。

替代原 webui.py 的 generate_prompt()，用结构化的实体/关系信息组装 prompt。
"""

from __future__ import annotations

from typing import List

from retriever import RetrievalResult


def build_prompt(query: str, retrieval: RetrievalResult) -> str:
    """根据检索结果组装最终送给 LLM 的 prompt。

    :param query: 用户原始问题
    :param retrieval: 双层检索结果
    :return: 完整 prompt 字符串
    """
    parts: list[str] = []

    # 指令
    parts.append(
        '<指令>你是一个医疗问答机器人，你需要根据给定的知识图谱信息回答用户的问题。'
        '请注意，你的全部回答必须完全基于给定的信息，不可自由发挥。'
        '如果根据给定信息无法给出答案，立刻回答“根据已知信息无法回答该问题”。</指令>'
    )
    parts.append(
        '<指令>请你仅针对医疗类问题提供简洁和专业的回答。'
        '如果问题不是医疗相关的，你一定要回答“我只能回答医疗相关的问题。”。</指令>'
    )

    # 实体信息
    if retrieval.entities:
        entity_sections: list[str] = []
        for ent in retrieval.entities:
            section = _format_entity(ent)
            if section:
                entity_sections.append(section)

        if entity_sections:
            parts.append("<知识图谱信息>")
            parts.append("\n\n".join(entity_sections))
            parts.append("</知识图谱信息>")
        else:
            parts.append("<知识图谱信息>未检索到相关信息。</知识图谱信息>")
    else:
        parts.append("<知识图谱信息>未检索到相关信息。</知识图谱信息>")

    # 用户问题
    parts.append(f"<用户问题>{query}</用户问题>")

    # 注意事项
    parts.append(
        '<注意>你要极其认真地判断上述信息中是否有用户问题所需的信息。'
        '如果没有相关信息，你必须直接回答"根据已知信息无法回答该问题"。</注意>'
    )
    parts.append(
        "<注意>你的回答必须完全基于上述<知识图谱信息>的内容，不可产生信息之外的答案。"
        "你起到的作用仅仅是整合信息的功能，不可以利用自身已经存在的知识进行回答。</注意>"
    )

    return "\n".join(parts)


def _format_entity(ent) -> str:
    """格式化单个实体的信息。"""
    lines: list[str] = []
    lines.append(f"【{ent.entity_type}】{ent.name}")

    # 属性
    if ent.attributes:
        for attr, value in ent.attributes.items():
            lines.append(f"  {attr}：{value}")

    # 邻居关系
    if ent.neighbors:
        for rel, items in ent.neighbors.items():
            if items:
                lines.append(f"  {rel}：{'、'.join(items)}")

    return "\n".join(lines)


def build_diagnostic_prompt(
    query: str,
    retrieval: RetrievalResult,
    symptoms: List[str],
    candidate_pool_dicts: List[dict],
    conversation_history: List[dict],
) -> str:
    """追问轮次的 prompt：引导 LLM 根据候选疾病和已收集症状提出有针对性的问题。

    与单轮 build_prompt() 的区别：
    - 包含已收集的症状列表（让 LLM 知道已经问过什么）
    - 包含候选疾病及依据（评分、关联症状、治疗方向、命中原因）
    - 包含对话历史（让 LLM 不重复已经问过的问题）
    - 指令侧重于"追问收集信息"而非"直接回答"

    参数：
        query: 用户本轮消息
        retrieval: 检索结果（prompt_entities 的详细信息）
        symptoms: 累积收集到的症状列表
        candidate_pool_dicts: 候选实体池（含完整评分、属性、邻居、来源）
        conversation_history: 已有对话历史 [{"role": "user"/"assistant", "content": "..."}]
    """
    parts: list[str] = []

    # 指令：告诉 LLM 角色和任务
    parts.append(
        '<指令>你是一个专业的医疗问诊助手。你的任务是通过追问收集用户的症状信息，'
        '帮助缩小可能的疾病范围。你只能基于给定的知识图谱信息进行分析。'
        '请提出具体、有针对性的问题，帮助区分候选疾病。'
        '如果问题不是医疗相关的，回答"我只能回答医疗相关的问题。"。</指令>'
    )

    # 已收集的症状：让 LLM 知道已经收集了哪些信息
    if symptoms:
        parts.append(f'<已收集的症状>{"、".join(symptoms)}</已收集的症状>')
    else:
        parts.append('<已收集的症状>暂无</已收集的症状>')

    # 候选疾病及依据：从 candidate_pool 中筛选疾病类型，按 rank_score 降序
    #   展示评分、关联症状、治疗方向、关联药品、命中原因
    #   让 LLM 理解为什么这些疾病是候选，以及如何通过追问区分它们
    diseases = [c for c in candidate_pool_dicts if c.get("entity_type") == "疾病"]
    diseases.sort(key=lambda d: d.get("scores", {}).get("rank_score", 0), reverse=True)
    if diseases:
        disease_sections: list[str] = []
        for i, d in enumerate(diseases[:5], 1):
            rank_score = d.get("scores", {}).get("rank_score", 0)
            section = f"{i}. {d['name']}（综合评分 {rank_score:.2f}）"
            neighbors = d.get("neighbors", {})
            symptoms_list = neighbors.get("疾病的症状", [])
            if symptoms_list:
                section += f"\n   关联症状：{'、'.join(symptoms_list[:8])}"
            treatments = neighbors.get("治疗的方法", [])
            if treatments:
                section += f"\n   治疗方向：{'、'.join(treatments)}"
            drugs = neighbors.get("疾病使用药品", [])
            if drugs:
                section += f"\n   关联药品：{'、'.join(drugs[:5])}"
            source = d.get("source", {})
            from_symptoms = source.get("from_symptoms", [])
            if from_symptoms:
                section += f"\n   命中原因：用户提到的{'、'.join(from_symptoms)}是该疾病的已知症状"
            disease_sections.append(section)
        parts.append("<候选疾病及依据>")
        parts.append("\n\n".join(disease_sections))
        parts.append("</候选疾病及依据>")

    # 知识图谱信息：prompt_entities 的详细属性和关系
    if retrieval.entities:
        entity_sections: list[str] = []
        for ent in retrieval.entities:
            section = _format_entity(ent)
            if section:
                entity_sections.append(section)
        if entity_sections:
            parts.append("<知识图谱信息>")
            parts.append("\n\n".join(entity_sections))
            parts.append("</知识图谱信息>")

    # 对话历史：让 LLM 避免重复已经问过的问题
    if conversation_history:
        parts.append("<对话历史>")
        for msg in conversation_history:
            role = "用户" if msg["role"] == "user" else "助手"
            parts.append(f"{role}：{msg['content']}")
        parts.append("</对话历史>")

    # 用户最新消息
    parts.append(f"<用户最新消息>{query}</用户最新消息>")

    # 注意事项：约束 LLM 行为
    parts.append(
        '<注意>你需要判断当前信息是否足以给出诊断建议。'
        '如果需要追问，请提出具体、有针对性的问题，帮助区分候选疾病。'
        '只能基于上述知识图谱信息，不可自由发挥。</注意>'
    )

    return "\n".join(parts)


def build_final_prompt(
    query: str,
    retrieval: RetrievalResult,
    symptoms: List[str],
    candidate_pool_dicts: List[dict],
    conversation_history: List[dict],
) -> str:
    """最终回答的 prompt：给出诊断分析和用药参考（不构成处方建议）。

    与追问 build_diagnostic_prompt() 的区别：
    - 指令侧重于"给出分析和建议"而非"追问收集信息"
    - 包含用药安全约束（不提供剂量、不说"你可以吃X"）
    - 展示疾病简介（intro），帮助 LLM 给出更完整的分析
    - 关联药品展示更多（8 个 vs 追问的 5 个）

    安全约束（写入指令中，不可绕过）：
    - 不提供剂量、疗程、用法用量
    - 不说"你可以吃X"或"建议你服用X"，只说"该疾病的关联药品包括..."
    - 免责声明由 qa_service.py 在生成完毕后追加（不在 prompt 中）

    参数同 build_diagnostic_prompt()。
    """
    parts: list[str] = []

    # 指令：包含用药安全约束
    parts.append(
        '<指令>你是一个专业的医疗问诊助手。根据已收集的症状信息和知识图谱数据，'
        '给出可能的诊断分析和参考信息。'
        '请注意：'
        '1. 你只能基于给定的知识图谱信息进行分析，不可自由发挥。'
        '2. 不要提供剂量、疗程、用法用量。'
        '3. 不要说"你可以吃X"或"建议你服用X"，只说"该疾病的关联药品包括..."。'
        '4. 如果知识图谱中没有相关信息，如实告知。</指令>'
    )

    # 已收集的症状
    if symptoms:
        parts.append(f'<已收集的症状>{"、".join(symptoms)}</已收集的症状>')

    # 候选疾病及依据：展示疾病简介（intro），帮助 LLM 给出完整分析
    diseases = [c for c in candidate_pool_dicts if c.get("entity_type") == "疾病"]
    diseases.sort(key=lambda d: d.get("scores", {}).get("rank_score", 0), reverse=True)
    if diseases:
        disease_sections: list[str] = []
        for i, d in enumerate(diseases[:5], 1):
            rank_score = d.get("scores", {}).get("rank_score", 0)
            section = f"{i}. {d['name']}（综合评分 {rank_score:.2f}）"
            attrs = d.get("attributes", {})
            intro = attrs.get("疾病简介", "")
            if intro:
                section += f"\n   简介：{intro[:200]}"
            neighbors = d.get("neighbors", {})
            symptoms_list = neighbors.get("疾病的症状", [])
            if symptoms_list:
                section += f"\n   关联症状：{'、'.join(symptoms_list[:8])}"
            treatments = neighbors.get("治疗的方法", [])
            if treatments:
                section += f"\n   治疗方向：{'、'.join(treatments)}"
            drugs = neighbors.get("疾病使用药品", [])
            if drugs:
                section += f"\n   关联药品：{'、'.join(drugs[:8])}"
            disease_sections.append(section)
        parts.append("<候选疾病及依据>")
        parts.append("\n\n".join(disease_sections))
        parts.append("</候选疾病及依据>")

    # 知识图谱信息
    if retrieval.entities:
        entity_sections: list[str] = []
        for ent in retrieval.entities:
            section = _format_entity(ent)
            if section:
                entity_sections.append(section)
        if entity_sections:
            parts.append("<知识图谱信息>")
            parts.append("\n\n".join(entity_sections))
            parts.append("</知识图谱信息>")

    # 对话历史
    if conversation_history:
        parts.append("<对话历史>")
        for msg in conversation_history:
            role = "用户" if msg["role"] == "user" else "助手"
            parts.append(f"{role}：{msg['content']}")
        parts.append("</对话历史>")

    # 用户最新消息
    parts.append(f"<用户最新消息>{query}</用户最新消息>")

    # 注意事项：用药安全约束
    parts.append(
        '<注意>基于上述知识图谱信息回答，不可自由发挥。'
        '不要提供剂量、疗程、用法用量。'
        '只列出知识图谱中关联的药品和治疗方向，不说"建议你吃X"。</注意>'
    )

    return "\n".join(parts)
