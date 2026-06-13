import asyncio
import json
import ner_model as ner
import openai
import random
import re
import logging
from typing import Dict, List, Optional, Tuple

from config import settings
from logging_setup import setup_logging
from kg_client import KGClient
from intent_router import execute_intents

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM 统一调用层：支持 ollama（本地）和 DashScope（云端）
# ---------------------------------------------------------------------------

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



def Intent_Recognition(query: str, model: str = "") -> str:
    """调用 LLM 做意图识别，返回 LLM 原始文本输出。

    :param query: 用户问题
    :param model: 模型名称，为空时使用配置默认值
    :return: LLM 输出文本，期望包含「查询XX」格式的意图标签列表，由
             :func:`intent_router.execute_intents` 解析。
    """
    prompt = f"""
阅读下列提示，回答问题（问题在输入的最后）:
当你试图识别用户问题中的查询意图时，你需要仔细分析问题，并在16个预定义的查询类别中一一进行判断。
对于每一个类别，思考用户的问题是否含有与该类别对应的意图。
如果判断用户的问题符合某个特定类别，就将该类别加入到输出列表中。
这样的方法要求你对每一个可能的查询意图进行系统性的考虑和评估，确保没有遗漏任何一个可能的分类。

**查询类别**
- "查询疾病简介"
- "查询疾病病因"
- "查询疾病预防措施"
- "查询疾病治疗周期"
- "查询治愈概率"
- "查询疾病易感人群"
- "查询疾病所需药品"
- "查询疾病宜吃食物"
- "查询疾病忌吃食物"
- "查询疾病所需检查项目"
- "查询疾病所属科目"
- "查询疾病的症状"
- "查询疾病的治疗方法"
- "查询疾病的并发疾病"
- "查询药品的生产商"

在处理用户的问题时，请按照以下步骤操作：
- 仔细阅读用户的问题。
- 对照上述查询类别列表，依次考虑每个类别是否与用户问题相关。
- 如果用户问题明确或隐含地包含了某个类别的查询意图，请将该类别的描述添加到输出列表中。
- 确保最终的输出列表包含了所有与用户问题相关的类别描述。

以下是一些含有隐晦性意图的例子，每个例子都采用了输入和输出格式，并包含了对你进行思维链形成的提示：
**示例1：**
输入："睡眠不好，这是为什么？"
输出：["查询疾病简介","查询疾病病因"]  # 这个问题隐含地询问了睡眠不好的病因
**示例2：**
输入："感冒了，怎么办才好？"
输出：["查询疾病简介","查询疾病所需药品", "查询疾病的治疗方法"]  # 用户可能既想知道应该吃哪些药品，也想了解治疗方法
**示例3：**
输入："跑步后膝盖痛，需要吃点什么？"
输出：["查询疾病简介","查询疾病宜吃食物", "查询疾病所需药品"]  # 这个问题可能既询问宜吃的食物，也可能在询问所需药品
**示例4：**
输入："我怎样才能避免冬天的流感和感冒？"
输出：["查询疾病简介","查询疾病预防措施"]  # 询问的是预防措施，但因为提到了两种疾病，这里隐含的是对共同预防措施的询问
**示例5：**
输入："头疼是什么原因，应该怎么办？"
输出：["查询疾病简介","查询疾病病因", "查询疾病的治疗方法"]  # 用户询问的是头疼的病因和治疗方法
**示例6：**
输入："如何知道自己是不是有艾滋病？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有艾滋病，一定一定要进行相关检查，这是根本性的！其次是查看疾病的病因，看看自己的行为是不是和病因重合。
**示例7：**
输入："我该怎么知道我自己是否得了21三体综合症呢？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有21三体综合症，一定一定要进行相关检查(比如染色体)，这是根本性的！其次是查看疾病的病因。
**示例8：**
输入："感冒了，怎么办？"
输出：["查询疾病简介","查询疾病的治疗方法","查询疾病所需药品","查询疾病所需检查项目","查询疾病宜吃食物"]  # 问怎么办，首选治疗方法。然后是要给用户推荐一些药，最后让他检查一下身体。同时，也推荐一下食物。
**示例9：**
输入："癌症会引发其他疾病吗？"
输出：["查询疾病简介","查询疾病的并发疾病","查询疾病简介"]  # 显然，用户问的是疾病并发疾病，随后可以给用户科普一下癌症简介。
**示例10：**
输入："葡萄糖浆的生产者是谁？葡萄糖浆是谁生产的？"
输出：["查询药品的生产商"]  # 显然，用户想要问药品的生产商

通过上述例子，我们希望你能够形成一套系统的思考过程，以准确识别出用户问题中的所有可能查询意图。请仔细分析用户的问题，考虑到其可能的多重含义，确保输出反映了所有相关的查询意图。

**注意：**
- 你的所有输出，都必须在上述**查询类别**范围内，不可创造新的名词与类别！
- 参考上述5个示例：在输出查询意图对应的列表之后，请紧跟着用"#"号开始的注释，简短地解释为什么选择这些意图选项。注释应当直接跟在列表后面，形成一条连续的输出。
- 你的输出的类别数量不应该超过5，如果确实有很多个，请你输出最有可能的5个！同时，你的解释不宜过长，但是得富有条理性。

现在，你已经知道如何解决问题了，请你解决下面这个问题并将结果输出！
问题输入："{query}"
输出的时候请确保输出内容都在**查询类别**中出现过。确保输出类别个数**不要超过5个**！确保你的解释和合乎逻辑的！注意，如果用户询问了有关疾病的问题，一般都要先介绍一下疾病，也就是有"查询疾病简介"这个需求。
再次检查你的输出都包含在**查询类别**:"查询疾病简介"、"查询疾病病因"、"查询疾病预防措施"、"查询疾病治疗周期"、"查询治愈概率"、"查询疾病易感人群"、"查询疾病所需药品"、"查询疾病宜吃食物"、"查询疾病忌吃食物"、"查询疾病所需检查项目"、"查询疾病所属科目"、"查询疾病的症状"、"查询疾病的治疗方法"、"查询疾病的并发疾病"、"查询药品的生产商"。
"""
    rec_result = llm_generate(prompt, model=model)
    logger.debug('意图识别结果: %s', rec_result)
    return rec_result


def _parse_query_expansion(raw_text: str) -> List[str]:
    """解析 LLM 输出的 JSON 数组，失败时返回空列表。"""
    match = re.search(r"\[[\s\S]*?\]", raw_text or "")
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("实体查询扩展 JSON 解析失败: %s", raw_text)
        return []
    if not isinstance(data, list):
        return []

    queries: List[str] = []
    seen: set[str] = set()
    for item in data:
        text = re.sub(r"[\s,，。;；、]+", "", str(item).strip())
        if not 2 <= len(text) <= 20 or text in seen:
            continue
        seen.add(text)
        queries.append(text)
        if len(queries) >= 5:
            break
    return queries


def expand_entity_queries(query: str, model: str = "") -> List[str]:
    """用 LLM 将口语化表达改写为 KG 实体查询词；只在实体主路径为空时调用。"""
    prompt = f"""
你是医疗知识图谱的实体查询改写器。
你的任务是把用户的口语化医疗表达改写成可能用于检索知识图谱的标准实体查询词。
要求：
1. 只输出 JSON 数组，不要解释。
2. 最多输出 5 个查询词。
3. 查询词可以是疾病、症状、检查项目或药品。
4. 不要诊断用户，不要给治疗建议。

示例：
用户：我脚崴了
输出：["踝关节扭伤", "踝扭伤", "足踝捩伤"]

用户：我眼皮红肿怎么办
输出：["眼睛红肿", "眼睑肿胀", "眼睑浮肿"]

用户：{query}
输出：
"""
    try:
        return _parse_query_expansion(llm_generate(prompt, model=model))
    except Exception:
        logger.exception("LLM 实体查询扩展失败")
        return []


async def get_entities_async(
    query: str,
    selected_model: str,
    bert_model,
    bert_tokenizer,
    rule,
    tfidf_r,
    device,
    idx2tag,
    status_placeholder=None,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    """异步实体检索：先跑主路径，主路径为空时再触发兜底扩展。"""
    base_entities = await asyncio.to_thread(
        ner.get_ner_result,
        bert_model,
        bert_tokenizer,
        query,
        rule,
        tfidf_r,
        device,
        idx2tag,
    )
    if base_entities:
        return base_entities, {
            "used_fallback": False,
            "fallback_queries": [],
            "source": "base",
        }

    if status_placeholder is not None:
        status_placeholder.info("基础实体识别未命中，正在进行兜底查询扩展...")

    expanded_queries = await asyncio.to_thread(expand_entity_queries, query, selected_model)
    return await asyncio.to_thread(
        ner.get_ner_result_with_fallback,
        bert_model,
        bert_tokenizer,
        query,
        rule,
        tfidf_r,
        device,
        idx2tag,
        expanded_queries,
        base_entities,
    )


async def analyze_query_async(
    query: str,
    selected_model: str,
    bert_model,
    bert_tokenizer,
    rule,
    tfidf_r,
    device,
    idx2tag,
    status_placeholder=None,
) -> Tuple[str, Dict[str, str], Dict[str, object]]:
    """并行执行意图识别与实体检索。"""
    intent_task = asyncio.create_task(
        asyncio.to_thread(Intent_Recognition, query, selected_model)
    )
    entity_task = asyncio.create_task(
        get_entities_async(
            query,
            selected_model,
            bert_model,
            bert_tokenizer,
            rule,
            tfidf_r,
            device,
            idx2tag,
            status_placeholder,
        )
    )
    intent_response, (entities, entity_meta) = await asyncio.gather(intent_task, entity_task)
    return intent_response, entities, entity_meta


def generate_prompt(
    response: str,
    query: str,
    client,
    bert_model,
    bert_tokenizer,
    rule,
    tfidf_r,
    device,
    idx2tag,
    entities: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, Dict[str, str]]:
    """根据 LLM 意图识别输出与 NER 抽取结果，组装最终送给 LLM 的 prompt。

    :return: ``(prompt, intents_str, entities)``；``intents_str`` 是「、」拼接的中文意图名。
    """
    if entities is None:
        entities = ner.get_ner_result(bert_model, bert_tokenizer, query, rule, tfidf_r, device, idx2tag)
    else:
        entities = dict(entities)
    # 统一封装为 KGClient（若调用方已传入 KGClient 则直接复用）
    kg = client if isinstance(client, KGClient) else KGClient(client)
    yitu: List[str] = []
    prompt = "<指令>你是一个医疗问答机器人，你需要根据给定的提示回答用户的问题。请注意，你的全部回答必须完全基于给定的提示，不可自由发挥。如果根据提示无法给出答案，立刻回答“根据已知信息无法回答该问题”。</指令>"
    prompt +="<指令>请你仅针对医疗类问题提供简洁和专业的回答。如果问题不是医疗相关的，你一定要回答“我只能回答医疗相关的问题。”，以明确告知你的回答限制。</指令>"
    if '疾病症状' in entities and  '疾病' not in entities:
        # 修复：原 client.run(...).data()[0] 在空结果时 IndexError
        res = kg.get_diseases_by_symptom(entities['疾病症状'])
        if len(res)>0:
            entities['疾病'] = random.choice(res)
            all_en = "、".join(res)
            prompt+=f"<提示>用户有{entities['疾病症状']}的情况，知识库推测其可能是得了{all_en}。请注意这只是一个推测，你需要明确告知用户这一点。</提示>"
    pre_len = len(prompt)
    # 用表驱动的意图路由替换原本 16 个重复 if 块；
    # intent_router 内部已修复「治疗周期 vs 治疗方法」子串误匹配 bug。
    intent_prompt, intent_names = execute_intents(response, entities, kg)
    prompt += intent_prompt
    yitu.extend(intent_names)
    if pre_len==len(prompt) :
        prompt += f"<提示>提示：知识库异常，没有相关信息！请你直接回答“根据已知信息无法回答该问题”！</提示>"
    prompt += f"<用户问题>{query}</用户问题>"
    prompt += f"<注意>现在你已经知道给定的“<提示></提示>”和“<用户问题></用户问题>”了,你要极其认真的判断提示里是否有用户问题所需的信息，如果没有相关信息，你必须直接回答“根据已知信息无法回答该问题”。</注意>"

    prompt += f"<注意>你一定要再次检查你的回答是否完全基于“<提示></提示>”的内容，不可产生提示之外的答案！换而言之，你的任务是根据用户的问题，将“<提示></提示>”整理成有条理、有逻辑的语句。你起到的作用仅仅是整合提示的功能，你一定不可以利用自身已经存在的知识进行回答，你必须从提示中找到问题的答案！</注意>"
    prompt += f"<注意>你必须充分的利用提示中的知识，不可将提示中的任何信息遗漏，你必须做到对提示信息的充分整合。你回答的任何一句话必须在提示中有所体现！如果根据提示无法给出答案，你必须回答“根据已知信息无法回答该问题”。<注意>"
    
    
    logger.debug('prompt: %s', prompt)
    return prompt,"、".join(yitu),entities


