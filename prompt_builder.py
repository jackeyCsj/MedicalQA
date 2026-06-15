"""LightRAG 风格的 Prompt 组装。

替代原 webui.py 的 generate_prompt()，用结构化的实体/关系信息组装 prompt。
"""

from __future__ import annotations

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
