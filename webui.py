"""LLM 调用层：提供 OpenAI 兼容的文本生成和流式聊天接口。"""

import openai
import logging

from config import settings
from logging_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _get_openai_client() -> openai.OpenAI:
    """返回当前 LLM_PROVIDER 的 OpenAI 兼容客户端。"""
    return openai.OpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )


def llm_generate(prompt: str, model: str = "") -> str:
    """调用 LLM 生成文本（非流式），返回完整响应字符串。

    :param prompt: 提示词
    :param model: 模型名称，为空时使用配置默认值
    """
    if settings.LLM_PROVIDER == "ollama":
        import ollama
        return ollama.generate(model=model or settings.OLLAMA_QWEN_MODEL, prompt=prompt)['response']
    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model or settings.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def llm_chat_stream(messages: list, model: str = ""):
    """调用 LLM 流式生成，yield 每个 chunk 的文本片段。

    :param messages: 消息列表
    :param model: 模型名称，为空时使用配置默认值
    """
    if settings.LLM_PROVIDER == "ollama":
        import ollama
        for chunk in ollama.chat(model=model or settings.OLLAMA_QWEN_MODEL, messages=messages, stream=True):
            yield chunk['message']['content']
        return
    client = _get_openai_client()
    stream = client.chat.completions.create(
        model=model or settings.LLM_MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
