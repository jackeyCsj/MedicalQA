"""LLM 关键词提取：替代 NER + 意图识别。

一次 LLM 调用同时提取：
- local_keywords：具体实体名（疾病、症状、药品、检查项目等）
- global_keywords：主题概念（治疗方法、预防措施、病因等）

参考 LightRAG 的 keyword extraction prompt 设计。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

_KEYWORD_PROMPT = """你是医疗知识图谱的关键词提取器。
从用户的医疗相关查询中提取两类关键词，用于知识图谱检索。

**提取规则**：
1. local_keywords（低层关键词）：提取查询中提到的具体医疗实体，如疾病名、症状名、药品名、检查项目名、身体部位等。
   - 如果用户描述的是症状而非疾病，同时提取症状和可能的身体部位/上下文
   - 例如"牙齿上长黑斑"→ 提取"牙齿"、"黑斑"
2. global_keywords（高层关键词）：提取查询隐含的主题/意图概念，如治疗方法、病因、预防、饮食等。
   - 根据查询的语义意图提取，而非字面匹配
   - 例如"怎么回事"→"病因"、"怎么办"→"治疗方法"

**输出格式**：只输出JSON，不要解释。
{{"local_keywords": ["关键词1", "关键词2"], "global_keywords": ["关键词1", "关键词2"]}}

**示例**：
用户：我牙齿上长黑斑怎么回事
输出：{{"local_keywords": ["牙齿", "黑斑", "牙齿黑斑"], "global_keywords": ["病因", "疾病简介"]}}

用户：高血压吃什么药好
输出：{{"local_keywords": ["高血压"], "global_keywords": ["治疗药品", "治疗方法"]}}

用户：感冒了怎么办
输出：{{"local_keywords": ["感冒"], "global_keywords": ["治疗方法", "治疗药品", "预防措施"]}}

用户：糖尿病的并发症有哪些
输出：{{"local_keywords": ["糖尿病"], "global_keywords": ["并发疾病"]}}

用户：{query}
输出："""


@dataclass
class Keywords:
    """关键词提取结果。"""
    local_keywords: List[str] = field(default_factory=list)
    global_keywords: List[str] = field(default_factory=list)


def extract_keywords(query: str, model: str = "") -> Keywords:
    """用 LLM 从用户查询中提取 local 和 global keywords。

    :param query: 用户问题
    :param model: LLM 模型名，为空时使用配置默认值
    :return: Keywords 对象
    """
    from webui import llm_generate

    prompt = _KEYWORD_PROMPT.format(query=query)
    try:
        raw = llm_generate(prompt, model=model)
        logger.debug("关键词提取原始输出: %s", raw)
        return _parse_keywords(raw)
    except Exception:
        logger.exception("关键词提取失败")
        return Keywords()


def _parse_keywords(raw: str) -> Keywords:
    """解析 LLM 输出的 JSON 关键词。"""
    # 尝试提取 JSON
    match = re.search(r"\{[\s\S]*?\}", raw or "")
    if not match:
        logger.warning("关键词 JSON 解析失败: %s", raw)
        return Keywords()

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("关键词 JSON 格式错误: %s", raw)
        return Keywords()

    local = _clean_keywords(data.get("local_keywords", []))
    global_kw = _clean_keywords(data.get("global_keywords", []))
    return Keywords(local_keywords=local, global_keywords=global_kw)


def _clean_keywords(items: list) -> List[str]:
    """清洗关键词列表：去重、去空、去标点。"""
    result: List[str] = []
    seen: set[str] = set()
    for item in items:
        text = re.sub(r"[][\s,，。.!！?？;；:：、\"'""''（）(){}<>《》]+", "", str(item).strip())
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
