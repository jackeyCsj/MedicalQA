from __future__ import annotations

import asyncio
import re
import threading
from typing import AsyncIterator, Dict, List, Optional

import py2neo

from config import settings
from kg_client import KGClient
from keyword_extractor import extract_keywords
from prompt_builder import build_prompt
from retriever import Retriever
from vector_index import VectorIndex
from webui import llm_chat_stream


class MedicalQAService:
    """Backend orchestration for keyword extraction, dual-level retrieval and answer streaming."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vi: Optional[VectorIndex] = None
        self._kg: Optional[KGClient] = None
        self._retriever: Optional[Retriever] = None

    def load(self) -> None:
        with self._lock:
            if self._vi is None:
                self._vi = VectorIndex.load()
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
        if self._retriever is None:
            self.load()
        assert self._retriever is not None
        return self._retriever

    def get_models(self) -> List[str]:
        if settings.LLM_PROVIDER == "ollama":
            return [settings.OLLAMA_QWEN_MODEL, settings.OLLAMA_LLAMA_MODEL]
        return [m.strip() for m in settings.LLM_MODELS.split(",") if m.strip()]

    async def stream_answer(
        self,
        message: str,
        model: str,
    ) -> AsyncIterator[tuple[str, str | Dict[str, object]]]:
        self.load()

        yield "status", "正在提取关键词..."
        keywords = await asyncio.to_thread(extract_keywords, message, model)
        yield "debug", {
            "step": "keywords",
            "local_keywords": keywords.local_keywords,
            "global_keywords": keywords.global_keywords,
        }

        yield "status", "正在检索知识图谱..."
        retrieval = await asyncio.to_thread(
            self.retriever.retrieve, keywords
        )
        entities_dict = {
            ent.name: {
                "type": ent.entity_type,
                "score": ent.score,
                "attributes": ent.attributes,
                "neighbors": ent.neighbors,
            }
            for ent in retrieval.entities
        }
        yield "debug", {"step": "retrieval", "entities": entities_dict}

        prompt = build_prompt(message, retrieval)
        knowledge_parts: List[str] = []
        for ent in retrieval.entities:
            lines = []
            for attr, value in ent.attributes.items():
                lines.append(f"{attr}：{value}")
            for rel, items in ent.neighbors.items():
                lines.append(f"{rel}：{'、'.join(items)}")
            if lines:
                knowledge_parts.append(f"[{ent.entity_type}]{ent.name}: {'; '.join(lines)}")
        knowledge = "\n".join(knowledge_parts)
        yield "debug", {
            "step": "prompt",
            "keywords": {
                "local": keywords.local_keywords,
                "global": keywords.global_keywords,
            },
            "knowledge": knowledge,
            "prompt": prompt,
        }

        yield "status", "正在生成回答..."
        async for chunk in self._stream_llm(prompt, model):
            if chunk:
                yield "token", chunk

        yield "done", {
            "entities": entities_dict,
            "keywords": {
                "local": keywords.local_keywords,
                "global": keywords.global_keywords,
            },
            "knowledge": knowledge,
        }

    async def _stream_llm(self, prompt: str, model: str) -> AsyncIterator[str]:
        queue: asyncio.Queue[tuple[str, Optional[str]]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for chunk in llm_chat_stream([{"role": "user", "content": prompt}], model=model):
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", chunk))
            except Exception as exc:
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
