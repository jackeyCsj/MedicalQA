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

from retrieval_intents import ALLOWED_GLOBAL_KEYWORDS, GLOBAL_KEYWORD_OPTIONS_TEXT

logger = logging.getLogger(__name__)

_KEYWORD_PROMPT = """你是医疗知识图谱的关键词提取器。
从用户的医疗相关查询中提取两类关键词，用于知识图谱检索。

**提取规则**：
0. 先在内部归纳用户主诉，把长句、口语化表达压缩成可检索的医学短语；不要输出这个归纳过程。
1. local_keywords（低层关键词）：提取最有检索价值的具体医疗实体，如疾病名、组合症状名、药品名、检查项目名等。
   - 用户描述症状时，优先输出"身体部位 + 症状/体征"的组合短语，再输出必要的规范化同义短语。
   - 不要只把组合症状拆成泛化碎片；例如"眼皮肿了并伴有疼痛"不要只输出"眼皮"、"肿胀"、"疼痛"。
   - 口语词要尽量规范化：如"眼皮"可对应"眼睑"，"肿了"可对应"肿胀"，"疼"可对应"疼痛/胀痛"。
   - local_keywords 按相关性从高到低排列，通常输出 3~6 个，避免过多泛化词。
   - 例如"牙齿上长黑斑"→ 提取"牙齿黑斑"、"牙齿长黑斑"、"黑斑"
2. global_keywords（高层关键词）：提取查询隐含的主题/意图概念。
   - 根据查询的语义意图提取，而非字面匹配
   - 例如"怎么回事"→"病因"、"怎么办"→"治疗方法"
   - 只能从以下候选意图中选择，不允许输出候选以外的词：{global_keyword_options}

**输出格式**：只输出JSON，不要解释。
{{"local_keywords": ["关键词1", "关键词2"], "global_keywords": ["关键词1", "关键词2"]}}

**示例**：
用户：我牙齿上长黑斑怎么回事
输出：{{"local_keywords": ["牙齿黑斑", "牙齿长黑斑", "黑斑"], "global_keywords": ["病因", "疾病简介"]}}

用户：我眼皮肿了并且还伴有疼痛是怎么回事
输出：{{"local_keywords": ["眼皮肿痛", "眼睑肿痛", "眼睑肿胀", "眼部胀痛"], "global_keywords": ["病因", "疾病简介"]}}

用户：高血压吃什么药好
输出：{{"local_keywords": ["高血压"], "global_keywords": ["治疗药品", "治疗方法"]}}

用户：感冒了怎么办
输出：{{"local_keywords": ["感冒"], "global_keywords": ["治疗方法", "治疗药品", "疾病简介"]}}

用户：我脚崴了有什么缓解疼痛的办法
输出：{{"local_keywords": ["脚崴", "踝关节扭伤", "踝部疼痛肿胀", "踝关节疼痛"], "global_keywords": ["治疗方法", "疾病简介"]}}

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

    prompt = _KEYWORD_PROMPT.format(
        query=query,
        global_keyword_options=GLOBAL_KEYWORD_OPTIONS_TEXT,
    )
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
    allowed_global = set(ALLOWED_GLOBAL_KEYWORDS)
    raw_global_kw = _clean_keywords(data.get("global_keywords", []))
    global_kw = [kw for kw in raw_global_kw if kw in allowed_global]
    dropped = [kw for kw in raw_global_kw if kw not in allowed_global]
    if dropped:
        logger.debug("丢弃不支持的 global_keywords: %s", dropped)
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
