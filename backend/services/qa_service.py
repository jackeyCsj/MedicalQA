from __future__ import annotations

import asyncio
import json
import os
import pickle
import re
import threading
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional

import py2neo
import torch
from transformers import BertTokenizer

import ner_model as ner
from config import settings
from intent_router import execute_intents
from kg_client import KGClient
from webui import (
    Intent_Recognition,
    analyze_query_async,
    expand_entity_queries,
    generate_prompt,
    llm_chat_stream,
)


@dataclass
class NERResources:
    bert_tokenizer: BertTokenizer
    bert_model: torch.nn.Module
    idx2tag: List[str]
    rule: object
    tfidf_r: object
    device: torch.device


class MedicalQAService:
    """Backend orchestration for intent recognition, KG retrieval and answer streaming."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._resources: Optional[NERResources] = None
        self._graph = None
        self._kg = None

    def load(self) -> None:
        with self._lock:
            if self._resources is None:
                self._resources = self._load_ner_resources()
            if self._kg is None:
                self._graph = py2neo.Graph(
                    settings.NEO4J_URL,
                    user=settings.NEO4J_USER,
                    password=settings.NEO4J_PASSWORD,
                    name=settings.NEO4J_DBNAME,
                )
                self._kg = KGClient(self._graph)

    def _load_ner_resources(self) -> NERResources:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        with open(os.path.join(settings.TMP_DIR, "tag2idx.npy"), "rb") as f:
            tag2idx = pickle.load(f)
        idx2tag = list(tag2idx)
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
        return NERResources(
            bert_tokenizer=tokenizer,
            bert_model=model,
            idx2tag=idx2tag,
            rule=rule,
            tfidf_r=tfidf_r,
            device=device,
        )

    @property
    def resources(self) -> NERResources:
        if self._resources is None:
            self.load()
        assert self._resources is not None
        return self._resources

    @property
    def kg(self) -> KGClient:
        if self._kg is None:
            self.load()
        assert self._kg is not None
        return self._kg

    def get_models(self) -> List[str]:
        if settings.LLM_PROVIDER == "dashscope":
            return [m.strip() for m in settings.DASHSCOPE_MODELS.split(",") if m.strip()]
        return [settings.OLLAMA_QWEN_MODEL, settings.OLLAMA_LLAMA_MODEL]

    async def analyze(self, message: str, model: str) -> tuple[str, Dict[str, str], Dict[str, object]]:
        self.load()
        resources = self.resources
        return await analyze_query_async(
            message,
            model,
            resources.bert_model,
            resources.bert_tokenizer,
            resources.rule,
            resources.tfidf_r,
            resources.device,
            resources.idx2tag,
            status_placeholder=None,
        )

    async def stream_answer(
        self,
        message: str,
        model: str,
    ) -> AsyncIterator[tuple[str, str | Dict[str, object]]]:
        self.load()
        resources = self.resources
        yield "status", "正在并行识别意图和抽取实体..."
        response, entities, entity_meta = await self.analyze(message, model)
        yield "debug", {"step": "entities", "entities": entities, "retrieval": entity_meta}
        yield "status", "正在查询知识图谱..."
        prompt, yitu, entities = generate_prompt(
            response,
            message,
            self.kg,
            resources.bert_model,
            resources.bert_tokenizer,
            resources.rule,
            resources.tfidf_r,
            resources.device,
            resources.idx2tag,
            entities=entities,
        )
        knowledge = re.findall(r"<提示>(.*?)</提示>", prompt)
        zhishiku_content = "\n".join([f"提示{idx + 1}, {kn}" for idx, kn in enumerate(knowledge) if len(kn) >= 3])
        yield "debug", {
            "step": "prompt",
            "intents": yitu,
            "knowledge": zhishiku_content,
            "prompt": prompt,
        }
        yield "status", "正在生成回答..."
        async for chunk in self._stream_llm(prompt, model):
            if chunk:
                yield "token", chunk
        yield "done", {"entities": entities, "intents": yitu, "knowledge": zhishiku_content}

    async def _stream_llm(self, prompt: str, model: str) -> AsyncIterator[str]:
        queue: asyncio.Queue[tuple[str, Optional[str]]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for chunk in llm_chat_stream([{"role": "user", "content": prompt}], model=model):
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", chunk))
            except Exception as exc:  # pragma: no cover - defensive path
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        threading.Thread(target=worker, daemon=True).start()
        while True:
            kind, value = await queue.get()
            if kind == "token" and value is not None:
                yield value
                continue
            if kind == "error":
                raise RuntimeError(value or "LLM streaming failed")
            break
