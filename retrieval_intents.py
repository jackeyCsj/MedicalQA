"""Shared retrieval intent definitions for LightRAG-style context pruning."""

from __future__ import annotations

from typing import Dict, List, Tuple


ALLOWED_GLOBAL_KEYWORDS = [
    "疾病简介",
    "病因",
    "预防措施",
    "治疗周期",
    "治愈概率",
    "易感人群",
    "治疗药品",
    "治疗方法",
    "宜吃食物",
    "忌吃食物",
    "检查项目",
    "并发疾病",
    "症状",
    "所属科目",
]

DEFAULT_GLOBAL_KEYWORDS = ["疾病简介"]

DISEASE_ATTRIBUTE_INTENT_MAP: Dict[str, str] = {
    "疾病简介": "疾病简介",
    "病因": "疾病病因",
    "预防措施": "预防措施",
    "治疗周期": "治疗周期",
    "治愈概率": "治愈概率",
    "易感人群": "疾病易感人群",
}

DISEASE_RELATION_INTENT_MAP: Dict[str, List[Tuple[str, str]]] = {
    "治疗药品": [("疾病使用药品", "药品")],
    "治疗方法": [("治疗的方法", "治疗方法")],
    "宜吃食物": [("疾病宜吃食物", "食物")],
    "忌吃食物": [("疾病忌吃食物", "食物")],
    "检查项目": [("疾病所需检查", "检查项目")],
    "并发疾病": [("疾病并发疾病", "疾病")],
    "症状": [("疾病的症状", "疾病症状")],
    "所属科目": [("疾病所属科目", "科目")],
}

GLOBAL_KEYWORD_OPTIONS_TEXT = "、".join(ALLOWED_GLOBAL_KEYWORDS)
