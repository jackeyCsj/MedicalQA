"""调试单条问答链路的中间输出。

用法示例：
- 只看 NER 和 TF-IDF，不调用大模型：
    python debug_pipeline.py "我睡觉起来左眼皮肿了，并且还有些疼痛是怎么回事" --probe 眼皮肿

    python debug_pipeline.py "我脚崴了，有什么缓解办法吗" --no-llm

    python debug_pipeline.py "我牙齿上长黑斑怎么回事"
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
from typing import Iterable

import py2neo
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer

import ner_model as ner
from config import settings
from intent_router import route_intents
from kg_client import KGClient
from webui import Intent_Recognition, expand_entity_queries, generate_prompt


def load_ner_resources():
    """加载 NER 推理所需资源，保持与 webui.load_model 的逻辑一致。"""
    with open(os.path.join(settings.TMP_DIR, "tag2idx.npy"), "rb") as f:
        tag2idx = pickle.load(f)
    idx2tag = list(tag2idx)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    rule = ner.rule_find()
    tfidf_r = ner.tfidf_alignment()
    tokenizer = BertTokenizer.from_pretrained(settings.NER_MODEL_NAME)
    model = ner.Bert_Model(settings.NER_MODEL_NAME, hidden_size=128, tag_num=len(tag2idx), bi=True)
    model.load_state_dict(
        torch.load(
            os.path.join(settings.MODEL_DIR, f"{settings.NER_CHECKPOINT}.pt"),
            map_location=device,
        )
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model, idx2tag, rule, tfidf_r, device


def inspect_ner(query: str, tokenizer, model, idx2tag, rule, tfidf_r, device) -> dict:
    """拆开 get_ner_result，打印每一步中间变量。"""
    sen_to = tokenizer.encode(query, add_special_tokens=True, return_tensors="pt").to(device)
    pred = model(sen_to).tolist()
    pred_tags = [idx2tag[i] for i in pred[1:-1]]

    model_result = ner.find_entities(pred_tags)
    model_result_word = []
    for begin, end, cls in model_result:
        model_result_word.append((begin, end, cls, query[begin : end + 1]))

    rule_result = rule.find(query)
    merge_result = ner.merge(model_result_word, rule_result)
    tfidf_result = tfidf_r.align(merge_result)

    return {
        "pred_tags": pred_tags,
        "model_result_word": model_result_word,
        "rule_result": rule_result,
        "merge_result": merge_result,
        "tfidf_result": tfidf_result,
    }


def print_section(title: str, value) -> None:
    print(f"\n==== {title} ====")
    print(value)


def print_tfidf_probe(tfidf_r, text: str, labels: Iterable[str], topk: int) -> None:
    """直接拿一段文本探测各实体类型下最像的标准实体。"""
    print(f"\n==== TF-IDF probe: {text} ====")
    for label in labels:
        if label not in tfidf_r.tag_2_tfidf_model:
            continue
        ent_emb = tfidf_r.tag_2_tfidf_model[label].transform([text])
        scores = cosine_similarity(ent_emb, tfidf_r.tag_2_embs[label])[0]
        ranked = scores.argsort()[::-1][:topk]
        candidates = [
            (tfidf_r.tag_2_entity[label][idx], round(float(scores[idx]), 4))
            for idx in ranked
        ]
        print(f"{label}: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(description="打印医疗问答 RAG 链路的中间输出")
    parser.add_argument("query", help="用户问题")
    parser.add_argument("--no-llm", action="store_true", help="不调用大模型，只调试 NER")
    parser.add_argument("--probe", default="", help="直接用某段文本探测 TF-IDF TopK 候选")
    parser.add_argument("--topk", type=int, default=5, help="TF-IDF probe 展示候选数量")
    args = parser.parse_args()

    tokenizer, model, idx2tag, rule, tfidf_r, device = load_ner_resources()
    ner_debug = inspect_ner(args.query, tokenizer, model, idx2tag, rule, tfidf_r, device)

    print_section("用户问题", args.query)
    print_section("模型 BIO 标签", list(zip(args.query, ner_debug["pred_tags"])))
    print_section("BERT+RNN 抽取实体", ner_debug["model_result_word"])
    print_section("Aho-Corasick 规则匹配实体", ner_debug["rule_result"])
    print_section("合并后的候选实体", ner_debug["merge_result"])
    print_section("TF-IDF 对齐后的标准实体", ner_debug["tfidf_result"])

    probe_text = args.probe.strip()
    if probe_text:
        print_tfidf_probe(tfidf_r, probe_text, ("疾病症状", "疾病", "检查项目"), args.topk)

    if args.no_llm:
        return

    graph = py2neo.Graph(
        settings.NEO4J_URL,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        name=settings.NEO4J_DBNAME,
    )
    kg = KGClient(graph)
    intent_response = Intent_Recognition(args.query)
    entities_for_prompt = ner_debug["tfidf_result"]
    entity_meta = {
        "used_fallback": False,
        "fallback_queries": [],
        "source": "base",
    }
    if not entities_for_prompt:
        llm_queries = expand_entity_queries(args.query)
        entities_for_prompt, entity_meta = ner.get_ner_result_with_fallback(
            model,
            tokenizer,
            args.query,
            rule,
            tfidf_r,
            device,
            idx2tag,
            expanded_queries=llm_queries,
            base_entities=entities_for_prompt,
        )
    routed = [spec.intent_name for spec in route_intents(intent_response)]
    prompt, yitu, entities = generate_prompt(
        intent_response,
        args.query,
        kg,
        model,
        tokenizer,
        rule,
        tfidf_r,
        device,
        idx2tag,
        entities=entities_for_prompt,
    )
    tips = re.findall(r"<提示>(.*?)</提示>", prompt)

    print_section("LLM 意图识别原始输出", intent_response)
    print_section("路由命中的意图", routed)
    print_section("实体兜底元信息", entity_meta)
    print_section("generate_prompt 使用的实体", entities)
    print_section("generate_prompt 返回的意图", yitu)
    print_section("知识库提示片段", tips)
    print_section("是否进入知识库异常兜底", "知识库异常" in prompt)


if __name__ == "__main__":
    main()
