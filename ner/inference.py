"""NER 推理流水线：规则匹配 + 模型预测 + 结果融合 + TF-IDF 实体对齐。

整体流程（:func:`get_ner_result`）：

1. BERT+RNN 模型预测 BIO 序列 → 得到模型实体集合
2. Aho-Corasick 自动机基于词典做规则匹配 → 得到规则实体集合
3. :func:`merge` 按实体长度优先级融合两组结果，去重叠
4. :class:`tfidf_alignment` 把抽取出的实体名对齐到知识图谱标准实体名

BIO 标签序列的定义如下：
    BIO 是一种标注方法，给每个字打一个标签，表示它在实体中的位置

    B-类型	Begin，     实体的开头第一个字	    一个实体的起点
    I-类型	Inside，    实体的中间/后续字	    实体的延续
    O	    Outside，   不属于任何实体	       普通字
    例子：
        字：    高      血      压    吃    什    么    药
        标签： B-疾病  I-疾病  I-疾病  O     O     O     O
        模型读完这串标签后，就知道：第 1~3 个字合起来是疾病实体 → "高血压"。

    为什么需要 B 和 I 两个标签？    因为要区分相邻的两个同类实体。比如：
    字：    高      血      压    和     糖      尿      病
    标签：  B-疾病 I-疾病 I-疾病   O    B-疾病  I-疾病  I-疾病
    如果没有 B/I 区分，全是 '疾病疾病疾病' O '疾病疾病疾病'，模型就不知道"高血压"和"糖尿病"是两个独立的实体。
    B 标签天然就是分隔符。

"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import ahocorasick
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .dataset import find_entities


FALLBACK_ENTITY_TYPES = ("疾病症状", "疾病", "检查项目", "药品")
_PUNCT_RE = re.compile(r"[\s,，。.!！?？;；:：、\"'“”‘’（）()【】\[\]{}<>《》]+")
_QUERY_STOP_WORDS = (
    "请问",
    "怎么办",
    "怎么回事",
    "怎么处理",
    "怎么治疗",
    "如何处理",
    "是什么原因",
    "什么原因",
    "有什么",
    "什么",
    "缓解办法",
    "缓解",
    "办法",
    "处理",
    "治疗",
    "需要",
    "可以",
    "能不能",
    "怎么",
    "如何",
    "有点",
    "有些",
    "一下",
    "自己",
    "我的",
    "我",
    "有",
    "了",
    "的",
    "吗",
    "呢",
    "啊",
    "呀",
    "吧",
)


def _dedupe_texts(texts: Sequence[str]) -> List[str]:
    """按出现顺序去重并去掉空字符串。"""
    result: List[str] = []
    seen: set[str] = set()
    for text in texts:
        item = str(text).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _compact_query(text: str) -> str:
    """去掉空白与标点，保留中文医学表达主体。"""
    return _PUNCT_RE.sub("", text or "")


def _strip_query_stop_words(text: str) -> str:
    """删除问句中高频但不利于实体召回的功能词。"""
    result = text
    for word in _QUERY_STOP_WORDS:
        result = result.replace(word, "")
    return result


def build_sliding_window_queries(
    text: str,
    min_len: int = 2,
    max_len: int = 6,
    limit: int = 20,
) -> List[str]:
    """从用户原句生成短语召回候选。

    只在主路径实体识别为空时作为兜底使用；按较长短语优先生成，避免过多噪声。
    """
    compacted = _compact_query(text)
    cleaned = _strip_query_stop_words(compacted)
    sources = _dedupe_texts([cleaned, compacted])
    candidates: List[str] = []
    for source in sources:
        if len(source) < min_len:
            continue
        upper = min(max_len, len(source))
        for size in range(upper, min_len - 1, -1):
            for start in range(0, len(source) - size + 1):
                candidates.append(source[start : start + size])
                if len(_dedupe_texts(candidates)) >= limit:
                    return _dedupe_texts(candidates)[:limit]
    return _dedupe_texts(candidates)[:limit]


# --------------------------------------------------------------------- rule


class rule_find:
    """基于 Aho-Corasick 自动机的多模式实体匹配器。

    词典文件位于 ``data/ent_aug/{type}.txt``，按行存储实体名。
    """

    def __init__(self) -> None:
        self.idx2type = idx2type = ["食物", "药品商", "治疗方法", "药品", "检查项目", "疾病", "疾病症状", "科目"]
        self.type2idx = type2idx = {t: i for i, t in enumerate(idx2type)}
        self.ahos = [ahocorasick.Automaton() for _ in range(len(self.type2idx))]

        for type in idx2type:
            with open(os.path.join('data', 'ent_aug', f'{type}.txt'), encoding='utf-8') as f:
                all_en = f.read().split('\n')
            for en in all_en:
                en = en.split(' ')[0]
                if len(en) >= 2:
                    self.ahos[type2idx[type]].add_word(en, en)
        for i in range(len(self.ahos)):
            self.ahos[i].make_automaton()

    def find(self, sen: str) -> List[Tuple[int, int, str, str]]:
        """对句子做规则匹配，返回 ``[(begin, end, type, word), ...]``。"""
        rule_result: List[Tuple[int, int, str, str]] = []
        mp: dict = {}
        all_res: list = []
        all_ty: list = []
        for i in range(len(self.ahos)):
            now = list(self.ahos[i].iter(sen))
            all_res.extend(now)
            for _ in range(len(now)):
                all_ty.append(self.idx2type[i])
        if len(all_res) != 0:
            all_res = sorted(all_res, key=lambda x: len(x[1]), reverse=True)
            for i, res in enumerate(all_res):
                be = res[0] - len(res[1]) + 1
                ed = res[0]
                if be in mp or ed in mp:
                    continue
                rule_result.append((be, ed, all_ty[i], res[1]))
                for t in range(be, ed + 1):
                    mp[t] = 1
        return rule_result


# --------------------------------------------------------------------- alignment


class tfidf_alignment:
    """把 NER 抽出的实体名通过 TF-IDF 余弦相似度对齐到 KG 标准实体名。"""

    def __init__(self) -> None:
        eneities_path = os.path.join('data', 'ent_aug')
        files = os.listdir(eneities_path)
        files = [docu for docu in files if '.py' not in docu]

        self.tag_2_embs: dict = {}
        self.tag_2_tfidf_model: dict = {}
        self.tag_2_entity: dict = {}
        for ty in files:
            with open(os.path.join(eneities_path, ty), 'r', encoding='utf-8') as f:
                entities = f.read().split('\n')
                entities = [ent for ent in entities if 1 <= len(ent.split(' ')[0]) <= 15]
                en_name = [ent.split(' ')[0] for ent in entities]
                ty = ty.strip('.txt')
                self.tag_2_entity[ty] = en_name
                tfidf_model = TfidfVectorizer(analyzer="char")
                embs = tfidf_model.fit_transform(en_name).toarray()
                self.tag_2_embs[ty] = embs
                self.tag_2_tfidf_model[ty] = tfidf_model

    def align(self, ent_list):
        """对齐实体；相似度阈值 ≥ 0.5 的才会被保留。返回 ``{type: 标准实体名}``。"""
        new_result: dict = {}
        for s, e, cls, ent in ent_list:
            ent_emb = self.tag_2_tfidf_model[cls].transform([ent])
            sim_score = cosine_similarity(ent_emb, self.tag_2_embs[cls])
            max_idx = sim_score[0].argmax()
            max_score = sim_score[0][max_idx]
            if max_score >= 0.5:
                new_result[cls] = self.tag_2_entity[cls][max_idx]
        return new_result

    def align_queries(
        self,
        queries: Sequence[str],
        labels: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, str]:
        """把一组查询词直接对齐到 KG 标准实体名。

        与 :meth:`align` 不同，这里没有 BIO 位置信息，主要用于主路径为空后的兜底召回。
        每个实体类型只保留相似度最高的一个候选。
        """
        result: Dict[str, str] = {}
        best_scores: Dict[str, float] = {}
        target_labels = labels or tuple(self.tag_2_tfidf_model.keys())
        for query in _dedupe_texts(queries):
            for label in target_labels:
                if label not in self.tag_2_tfidf_model or label not in self.tag_2_embs:
                    continue
                ent_emb = self.tag_2_tfidf_model[label].transform([query])
                sim_score = cosine_similarity(ent_emb, self.tag_2_embs[label])
                max_idx = sim_score[0].argmax()
                max_score = float(sim_score[0][max_idx])
                if max_score < threshold or max_score <= best_scores.get(label, -1.0):
                    continue
                result[label] = self.tag_2_entity[label][max_idx]
                best_scores[label] = max_score
        return result


# --------------------------------------------------------------------- merge


def merge(model_result_word, rule_result):
    """按 `word` 长度降序合并模型与规则两组结果，去除位置重叠。"""
    result = model_result_word + rule_result
    result = sorted(result, key=lambda x: len(x[-1]), reverse=True)
    check_result = []
    mp: dict = {}
    for res in result:
        if res[0] in mp or res[1] in mp:
            continue
        check_result.append(res)
        for i in range(res[0], res[1] + 1):
            mp[i] = 1
    return check_result


# --------------------------------------------------------------------- entry


def get_ner_result(model, tokenizer, sen, rule, tfidf_r, device, idx2tag):
    """NER 推理入口；返回 ``{type: 标准实体名}`` 字典。"""
    sen_to = tokenizer.encode(sen, add_special_tokens=True, return_tensors='pt').to(device)
    pre = model(sen_to).tolist()

    pre_tag = [idx2tag[i] for i in pre[1:-1]]
    model_result = find_entities(pre_tag)
    model_result_word = []
    for res in model_result:
        word = sen[res[0]:res[1] + 1]
        model_result_word.append((res[0], res[1], res[2], word))
    rule_result = rule.find(sen)

    merge_result = merge(model_result_word, rule_result)
    tfidf_result = tfidf_r.align(merge_result)
    return tfidf_result


def get_ner_result_with_fallback(
    model,
    tokenizer,
    sen,
    rule,
    tfidf_r,
    device,
    idx2tag,
    expanded_queries: Optional[Sequence[str]] = None,
    base_entities: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    """实体识别高层入口：主路径为空时才启用短语/LLM 查询词兜底。

    ``expanded_queries`` 由调用方提供，通常来自 LLM 多查询扩展；本函数只负责
    把滑动窗口短语与扩展查询词合并后对齐到 KG 标准实体。
    """
    entities = base_entities
    if entities is None:
        entities = get_ner_result(model, tokenizer, sen, rule, tfidf_r, device, idx2tag)

    if entities:
        return entities, {
            "used_fallback": False,
            "fallback_queries": [],
            "source": "base",
        }

    window_queries = build_sliding_window_queries(sen)
    fallback_queries = _dedupe_texts(window_queries + list(expanded_queries or []))
    fallback_entities = tfidf_r.align_queries(fallback_queries, labels=FALLBACK_ENTITY_TYPES)
    return fallback_entities, {
        "used_fallback": True,
        "fallback_queries": fallback_queries,
        "source": "fallback",
    }
