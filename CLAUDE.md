# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

RAGQnASystem — 基于 LightRAG 范式的医疗智能问答系统。

- Neo4j 医疗知识图谱（4.4 万实体、31 万关系）做精确检索，缓解 LLM 幻觉
- LLM 关键词提取（替代 BERT NER）+ FAISS 向量语义检索 + 图谱一跳邻居扩展
- 参考 LightRAG 双层检索范式：low-level（具体实体匹配）+ high-level（主题关系补充）
- 支持多 LLM 提供商：DashScope、MiMo、DeepSeek、Ollama，通过 `LLM_PROVIDER` 切换
- 前后端分离：FastAPI + Vue3 + PostgreSQL

## 常用命令

```bash
# --- 新版前后端分离入口 ---

# 后端（FastAPI）
python -m uvicorn backend.main:app --reload

# 前端（Vue3 + Vite）
cd frontend && npm.cmd install && npm.cmd run dev   # Windows PowerShell 用 npm.cmd

# --- 构建知识图谱（会询问是否删除 Neo4j 中已有数据）---
python build_up_graph.py --website http://localhost:7474 --user neo4j --password <密码> --dbname neo4j

# --- 构建向量索引（首次使用前必须执行，约几分钟）---
python build_vector_index.py

# --- 调试流水线 ---
python debug_pipeline.py "我牙齿上长黑斑怎么回事"

# --- 数据库迁移 ---
alembic upgrade head            # 应用数据库迁移（backend/alembic/）

# --- 测试 ---
pytest tests/
```

## 架构设计

```
frontend/（Vue3+Vite+TS）──fetch/SSE──→ backend/（FastAPI）
                                            │
                              ┌──────────────┼──────────────┐
                              ▼              ▼              ▼
                     routers/auth.py  routers/chat.py  services/auth_service.py
                                            │
                                            ▼
                                   services/qa_service.py
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
            keyword_extractor.py      retriever.py          prompt_builder.py
           （LLM 关键词提取）       （双层检索器）         （Prompt 组装）
                    │                       │
                    ▼                       ▼
              webui.py               vector_index.py
           （LLM 调用层）           （FAISS 向量索引）
                    │                       │
                    ▼                       ▼
              config.py               kg_client.py
           （集中配置）            （Neo4j Cypher 查询）
```

**数据流（LightRAG 范式）：**
1. `qa_service.py:stream_answer()` 接收用户消息
2. `keyword_extractor.extract_keywords()` — 1 次 LLM 调用，提取 local_keywords（具体实体）和 global_keywords（主题概念）
3. `retriever.Retriever.retrieve()` — 双层检索：
   - **Low-level**：local_keywords → FAISS 向量匹配 Top-K 实体 → Neo4j 查询属性和一跳邻居
   - **High-level**：global_keywords → 对已匹配疾病实体补充目标关系（如"治疗方法"→ 治疗的方法）
4. `prompt_builder.build_prompt()` — LightRAG 风格 prompt：`<指令>` + `<知识图谱信息>` + `<用户问题>` + `<注意>`
5. LLM 流式生成回答，通过 SSE（`status`/`debug`/`token`/`done`/`error` 事件）推送到前端

**关键设计决策：**
- LLM 关键词提取替代 BERT NER + 意图识别（1 次调用替代 2 次，成本更低，准确率更高）
- FAISS 向量索引（text2vec-base-chinese，768 维）替代 TF-IDF 对齐 + Aho-Corasick 规则匹配
- 双层检索参考 LightRAG：low-level 精确实体 + high-level 主题补充
- LLM 被严格约束为只能基于 `<知识图谱信息>` 内容回答，不可自由发挥
- `ner/` 目录保留训练代码，但运行时不再加载（已从 qa_service 中移除）

## 配置说明

所有配置通过 `config.py` 的 `Settings` dataclass 管理，支持环境变量覆盖。

**LLM 提供商**：通过 `LLM_PROVIDER` 切换，内置预设表（`_LLM_PRESETS`）自动填充 base_url/model/models：

| 提供商 | `LLM_PROVIDER` | API Key 环境变量 | 默认模型 |
|--------|----------------|------------------|----------|
| 阿里百炼 | `dashscope` | `DASHSCOPE_API_KEY` | `qwen-plus-latest` |
| 小米 MiMo | `mimo` | `MIMO_API_KEY` | `mimo-v2.5-pro` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-v4-pro` |
| Ollama 本地 | `ollama` | 无需 | `qwen:32b` |

**全部配置项：**

| 变量名 | 默认值 | 用途 |
|--------|--------|------|
| `NEO4J_URL` | `bolt://localhost:7687` | Neo4j 连接地址 |
| `NEO4J_USER` | `neo4j` | Neo4j 用户名 |
| `NEO4J_PASSWORD` | *（请自行设置）* | Neo4j 密码 |
| `NEO4J_DBNAME` | `neo4j` | 数据库名 |
| `LLM_PROVIDER` | `dashscope` | LLM 后端提供商 |
| `LLM_API_KEY` | *（空）* | 通用 API Key（覆盖提供商特定 key） |
| `LLM_BASE_URL` | *（自动）* | API 地址（通常由预设表自动填充） |
| `LLM_MODEL` | *（自动）* | 默认模型名（通常由预设表自动填充） |
| `LLM_MODELS` | *（自动）* | 可选模型列表（逗号分隔） |
| `OLLAMA_QWEN_MODEL` | `qwen:32b` | Ollama Qwen 模型标签 |
| `OLLAMA_LLAMA_MODEL` | `llama2-chinese:13b-chat-q8_0` | Ollama Llama2 模型标签 |
| `DATA_DIR` | `data` | 数据目录 |
| `TMP_DIR` | `tmp_data` | 缓存目录 |
| `MODEL_DIR` | `model` | 模型权重目录 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `DATABASE_URL` | `postgresql+psycopg://...localhost:5432/medicalqa` | PostgreSQL 连接串 |
| `SESSION_SECRET` | `dev-insecure-change-me` | Session 密钥，部署时必须修改 |
| `SESSION_COOKIE_NAME` | `session` | Cookie 名称 |
| `SESSION_EXPIRE_DAYS` | `7` | 登录会话有效期（天） |
| `COOKIE_SECURE` | `False` | 生产 HTTPS 环境设为 True |
| `CORS_ORIGINS` | `http://localhost:5173` | 允许跨域的前端源 |

建议在项目根目录创建 `.env` 文件（已在 `.gitignore`），启动前执行：`export $(cat .env | xargs)`

## 核心模块

### 关键词提取（`keyword_extractor.py`）
- 1 次 LLM 调用同时提取 local_keywords（具体实体名）和 global_keywords（主题概念）
- 替代原 BERT NER + LLM 意图识别（2 次调用），成本更低

### 向量索引（`vector_index.py`）
- FAISS IndexFlatIP，使用 `shibing624/text2vec-base-chinese`（768 维）生成 embedding
- 索引文件：`tmp_data/vector_index/entities.faiss` + `entities_meta.npz`
- 构建脚本：`python build_vector_index.py`（一次性，约几分钟，49278 条实体）

### 双层检索器（`retriever.py`）
- **Low-level**：local_keywords → FAISS 向量匹配 → Neo4j 查询属性和一跳邻居
- **High-level**：global_keywords → 对疾病实体补充目标关系（药品/食物/检查等）

### Prompt 组装（`prompt_builder.py`）
- LightRAG 风格：`<指令>` + `<知识图谱信息>`（实体属性+关系）+ `<用户问题>` + `<注意>`

### LLM 调用层（`webui.py`）
- `llm_generate()`：非流式生成
- `llm_chat_stream()`：流式生成（yield chunk）
- 支持 OpenAI 兼容 API + Ollama 本地调用

### NER 模块（`ner/`，已从运行时移除，保留训练代码）
- 训练入口：`python -m ner.train`
- 保存权重：`model/best_roberta_rnn_model_ent_aug.pt`

## 知识图谱

8 类实体（44,656 节点）：疾病(8808)、药品(3828)、食物(4870)、检查项目(3353)、科目(54)、药品商(17201)、疾病症状(5998)、治疗方法(544)

7 类疾病属性：疾病简介、疾病病因、预防措施、治疗周期、治愈概率、疾病易感人群、名称

11 类关系（312,159 条边）：疾病使用药品、疾病宜吃食物、疾病忌吃食物、疾病所需检查、疾病所属科目、疾病的症状、治疗的方法、疾病并发疾病、药品商生产药品 等

## 后端 API（`backend/`）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/register` | 注册（用户名 3-64 字符，密码 8-128 字符） |
| POST | `/api/auth/login` | 登录，设置 HttpOnly session cookie |
| POST | `/api/auth/logout` | 登出，撤销 session |
| GET | `/api/auth/me` | 当前用户信息，未登录返回 401 |
| GET | `/api/models` | 可选 LLM 模型列表 |
| POST | `/api/chat/stream` | SSE 流式问答，事件：`status`/`debug`/`token`/`done`/`error` |

认证：Argon2 密码哈希 + 服务端 session（SHA-256 hash 存库，cookie 存随机 token）+ 登录限流（5 次/10 分钟）

## 启动前自检

| 检查项 | 验证方法 |
|--------|----------|
| Neo4j 运行中 | 浏览器 http://localhost:7474 能看到控制台 |
| 知识图谱已导入 | `MATCH (n) RETURN count(n)` 返回 ~44656 |
| 向量索引已构建 | `tmp_data/vector_index/entities.faiss` 存在（运行 `python build_vector_index.py`） |
| PostgreSQL 运行中 | `psql` 能连接，`medicalqa` 数据库已创建 |
| 数据库迁移已应用 | `alembic upgrade head` 成功 |
| LLM 可用 | `.env` 中对应提供商的 API Key 已设置；Ollama：`ollama list` 有模型 |

## Behavioral Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
