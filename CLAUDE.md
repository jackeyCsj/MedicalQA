# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

RAGQnASystem — 基于知识图谱 RAG + 大语言模型的医疗智能问答系统。

- Neo4j 医疗知识图谱（4.4 万实体、31 万关系）做精确检索，缓解 LLM 幻觉
- BERT+RNN NER（F1 97.40%）+ Aho-Corasick 规则匹配 + TF-IDF 对齐
- 支持 DashScope（云端）和 Ollama（本地）两种 LLM 后端
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

# --- NER ---
python ner_data.py              # 生成训练数据（需要先构建知识图谱）
python -m ner.train             # 训练 NER 模型

# --- 调试流水线 ---
python debug_pipeline.py "我脚崴了，有什么缓解办法吗"

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
              webui.py              intent_router.py          kg_client.py
           （核心 QA 逻辑）          （表驱动意图路由）        （Neo4j Cypher 查询）
                    │
                    ▼
              ner/（NER 子包）
                    │
                    ▼
              config.py（集中配置）
```

**数据流：**
1. `qa_service.py:stream_answer()` 接收用户消息
2. NER 实体抽取与 LLM 意图识别**并行执行**（`webui.py:analyze_query_async`，asyncio.gather）
3. `intent_router.py:route_intents()` 按关键字长度降序匹配，避免子串冲突
4. `kg_client.py:KGClient` 执行 Cypher 查询
5. 主路径实体为空时触发 **fallback**：短语召回 + LLM 查询扩展（最多 5 个查询词）
6. `webui.py:generate_prompt()` 拼装 `<指令>/<提示>/<用户问题>/<注意>` prompt
7. LLM 流式生成回答，通过 SSE（`status`/`debug`/`token`/`done`/`error` 事件）推送到前端

**关键设计决策：**
- 表驱动意图路由（`INTENT_SPECS`）替代 16 个 if-else 块
- NER 双重策略：BERT 模型预测 + Aho-Corasick 规则匹配，按实体长度优先级合并
- TF-IDF 对齐将候选实体映射到知识图谱标准实体名（相似度 ≥ 0.5）
- LLM 被严格约束为只能基于知识图谱 `<提示>` 内容回答，不可自由发挥
- `qa_service.py` 不重写核心逻辑，直接从 `webui.py` 导入共享函数

## 配置说明

所有配置通过 `config.py` 的 `Settings` dataclass 管理，支持环境变量覆盖：

| 变量名 | 默认值 | 用途 |
|--------|--------|------|
| `NEO4J_URL` | `bolt://localhost:7687` | Neo4j 连接地址 |
| `NEO4J_USER` | `neo4j` | Neo4j 用户名 |
| `NEO4J_PASSWORD` | *（请自行设置）* | Neo4j 密码 |
| `NEO4J_DBNAME` | `neo4j` | 数据库名 |
| `LLM_PROVIDER` | `dashscope` | LLM 后端：`dashscope`（云端）或 `ollama`（本地） |
| `DASHSCOPE_API_KEY` | *（空）* | 阿里百炼 API Key |
| `DASHSCOPE_MODEL` | `qwen-plus-latest` | 百炼模型 ID |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 百炼兼容接口 |
| `DASHSCOPE_MODELS` | `qwen3.7-plus,qwen3.7-max,...` | 可选模型列表（逗号分隔） |
| `OLLAMA_QWEN_MODEL` | `qwen:32b` | Ollama Qwen 模型标签 |
| `OLLAMA_LLAMA_MODEL` | `llama2-chinese:13b-chat-q8_0` | Ollama Llama2 模型标签 |
| `NER_MODEL_NAME` | `model/chinese-roberta-wwm-ext` | BERT 基座路径 |
| `NER_CHECKPOINT` | `best_roberta_rnn_model_ent_aug` | NER 权重文件名（不含 .pt） |
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

## NER 模块（`ner/`）

- **model.py**：`Bert_Model` — BERT → 双向 RNN（`self.gru` 是历史命名，实为 `nn.RNN`）→ 线性分类器 → BIO 标签
- **dataset.py**：`Entity_Extend`，3 种数据增强（实体替换/掩码/拼接）；epoch ≥ 7 奇数轮触发
- **inference.py**：`get_ner_result()` 模型预测 + 规则匹配 + 合并 + TF-IDF 对齐；`get_ner_result_with_fallback()` 主路径为空时触发短语召回 + LLM 扩展
- **train.py**：训练入口，保存 `model/best_roberta_rnn_model_ent_aug.pt` 和 `tmp_data/tag2idx.npy`
- **ner_model.py**（根目录）：向后兼容 shim，重新导出 `ner/` 的公共 API

推理依赖文件：`model/chinese-roberta-wwm-ext/`、`model/best_roberta_rnn_model_ent_aug.pt`、`tmp_data/tag2idx.npy`

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
| BERT base 模型 | `model/chinese-roberta-wwm-ext/` 下有 config.json + vocab.txt |
| NER 微调权重 | `model/best_roberta_rnn_model_ent_aug.pt` 存在 |
| tag2idx 映射 | `tmp_data/tag2idx.npy` 存在 |
| PostgreSQL 运行中 | `psql` 能连接，`medicalqa` 数据库已创建 |
| 数据库迁移已应用 | `alembic upgrade head` 成功 |
| LLM 可用 | DashScope：`DASHSCOPE_API_KEY` 已设置；Ollama：`ollama list` 有模型 |

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
