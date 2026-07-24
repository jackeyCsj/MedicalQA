<div align="center">

# 🏥 RAGQnASystem / MedicalQA

### 基于 Neo4j 知识图谱 + FAISS 向量检索 + LLM 的多轮诊断式医疗问答系统

<br>

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Vue 3](https://img.shields.io/badge/Vue-3.x-42B883?logo=vuedotjs&logoColor=white)](https://vuejs.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-JSONB-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-KG-4581C3?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br>

**多轮问诊** · **知识图谱 RAG** · **FAISS 语义召回** · **SSE 流式回答** · **用药安全约束**

<br>

<img src="img/overall.png" width="100%" />

<sub>说明：图片暂未随新版架构同步，当前 README 以实际代码中的 FastAPI + Vue3 + PostgreSQL 多轮诊断流程为准。</sub>

<br>

[快速开始](#-快速开始) · [系统流程](#-系统流程) · [核心模块](#-核心模块) · [配置说明](#-配置说明) · [API](#-后端-api)

</div>

---

## 💡 项目简介

本项目是一个医疗智能问答系统，当前主线已经从早期的 Streamlit + NER 单轮问答，演进为 **FastAPI + Vue3 + PostgreSQL** 的前后端分离应用。系统通过 Neo4j 医疗知识图谱提供结构化事实，通过 FAISS 向量索引做实体语义召回，再由 LLM 完成关键词提取、追问生成和最终回答组织。

核心目标不是让大模型自由诊断，而是让它在受控知识图谱信息内进行整理和表达：

- **Neo4j 知识图谱**：约 4.4 万实体、31 万关系，覆盖疾病、症状、药品、食物、检查、科目、治疗方法、药品商等。
- **FAISS 向量检索**：使用 <code>shibing624/text2vec-base-chinese</code> 为实体名建立语义索引，替代运行时 BERT NER 主链路。
- **LLM 关键词提取**：一次调用同时提取 <code>local_keywords</code>（具体实体/症状）和 <code>global_keywords</code>（疾病简介、病因、治疗药品、检查项目等意图主题）。
- **多轮诊断状态机**：首轮建立候选池，后续轮次按新症状重评分，规则优先决定继续追问还是给出最终回答。
- **医疗安全约束**：最终回答不提供剂量、疗程、用法用量，不输出“你可以吃 X”这类处方式建议，并强制追加免责声明。

> 数据集来源：[Open-KG 疾病信息数据集](http://data.openkg.cn/dataset/disease-information)
> 参考项目：[RAGOnMedicalKG](https://github.com/liuhuanyong/RAGOnMedicalKG) · [QASystemOnMedicalKG](https://github.com/liuhuanyong/QASystemOnMedicalKG)

---

## ✨ 核心特性

| 能力 | 当前实现 |
|---|---|
| 知识图谱检索 | <code>kg_client.py</code> 封装 Neo4j Cypher 查询，补充疾病属性和一跳邻居关系 |
| 双层 RAG | <code>retriever.py</code> 同时做 low-level 实体召回和 high-level 主题关系扩展 |
| 向量索引 | <code>vector_index.py</code> 使用 FAISS <code>IndexFlatIP</code>，实体元数据保存到 <code>tmp_data/vector_index/</code> |
| 多轮诊断 | <code>backend/services/qa_service.py</code> 编排首轮、后续轮次、单轮兼容三条路径 |
| 路由决策 | <code>backend/services/diagnostic.py</code> 根据轮次、候选疾病数量、评分断层、症状数量决定 <code>ask</code> / <code>answer</code> |
| 会话持久化 | PostgreSQL 保存用户、Session、会话、消息和每轮 RAG 快照 |
| 流式交互 | 后端 <code>/api/chat/stream</code> 输出 SSE，前端逐 token 渲染 Markdown |
| 多模型支持 | <code>LLM_PROVIDER</code> 支持 <code>dashscope</code> / <code>mimo</code> / <code>deepseek</code> / <code>ollama</code>，统一走 OpenAI 兼容接口或 Ollama 本地调用 |

> <code>ner/</code> 目录和 <code>ner_data.py</code> 仍保留训练代码，但当前运行时主链路已经不再加载 BERT NER 模型。

---

## 🚀 快速开始

### 1. 准备环境

建议使用 Python 3.10。前端需要 Node.js 18+。

~~~bash
git clone https://github.com/honeyandme/RAGQnASystem.git
cd RAGQnASystem

conda create -n medicalqa python=3.10 -y
conda activate medicalqa
pip install -r requirements.txt
~~~

当前代码还直接依赖 <code>faiss</code> 和 <code>sentence-transformers</code>。如果安装依赖后仍提示 <code>No module named faiss</code> 或 <code>No module named sentence_transformers</code>，请额外安装：

~~~bash
pip install sentence-transformers faiss-cpu
~~~

如果 Windows 下 <code>faiss-cpu</code> 安装失败，可以改用 Conda 渠道安装 FAISS：

~~~bash
conda install -c pytorch faiss-cpu
~~~

---

### 2. 配置环境变量

复制模板并修改本地配置：

~~~bash
cp .env.example .env
~~~

Windows PowerShell 可使用：

~~~powershell
Copy-Item .env.example .env
~~~

至少需要确认以下几类配置：

| 配置 | 说明 |
|---|---|
| <code>NEO4J_URL</code> / <code>NEO4J_USER</code> / <code>NEO4J_PASSWORD</code> / <code>NEO4J_DBNAME</code> | Neo4j 连接信息，默认推荐 <code>bolt://localhost:7687</code> |
| <code>DATABASE_URL</code> | PostgreSQL 连接串，例如 <code>postgresql+psycopg://postgres:postgres@localhost:5431/medicalqa</code> |
| <code>LLM_PROVIDER</code> | 可选 <code>dashscope</code>、<code>mimo</code>、<code>deepseek</code>、<code>ollama</code> |
| 厂商 API Key | 如 <code>DASHSCOPE_API_KEY</code>、<code>MIMO_API_KEY</code>、<code>DEEPSEEK_API_KEY</code>；Ollama 不需要 |
| <code>SESSION_SECRET</code> | 生产环境必须改成高强度随机值 |
| <code>MAX_CLARIFICATION_TURNS</code> | 最大追问轮次，默认 <code>2</code> |
| <code>CLARIFICATION_ENABLED</code> | 是否启用多轮诊断，默认 <code>True</code> |

> <code>.env</code> 已被 Git 忽略，不要把真实 API Key、数据库密码提交到仓库。

---

### 3. 启动并构建 Neo4j 知识图谱

确认 Neo4j 正在运行后，执行：

~~~bash
python build_up_graph.py
~~~

也可以显式传参：

~~~bash
python build_up_graph.py \
  --website bolt://localhost:7687 \
  --user neo4j \
  --password <你的Neo4j密码> \
  --dbname neo4j
~~~

<code>build_up_graph.py</code> 会读取 <code>data/medical_new_2.json</code>，创建 8 类实体和 11 类关系，并更新 <code>data/ent_aug/*.txt</code> 实体词表。运行时脚本会询问是否删除 Neo4j 中已有数据，请确认目标库可被重建后再输入确认。

构建完成后可在 Neo4j Browser 中验证：

~~~cypher
MATCH (n) RETURN count(n);
MATCH ()-[r]->() RETURN count(r);
MATCH (a:疾病 {名称:'高血压'}) RETURN a.疾病简介;
~~~

预期规模约为 44,656 个节点、312,159 条关系。

---

### 4. 构建 FAISS 向量索引

首次运行前必须构建实体向量索引：

~~~bash
python build_vector_index.py
~~~

脚本会从 <code>data/ent_aug/*.txt</code> 读取实体名，用 <code>shibing624/text2vec-base-chinese</code> 生成 embedding，并保存：

~~~text
tmp_data/vector_index/entities.faiss
tmp_data/vector_index/entities_meta.npz
~~~

注意：运行时 <code>VectorIndex</code> 以 <code>local_files_only=True</code> 加载 embedding 模型。因此首次构建索引时需要让模型成功下载并缓存到本机，或提前把模型放入本地 HuggingFace 缓存。

---

### 5. 准备 PostgreSQL 并执行迁移

创建数据库后执行 Alembic 迁移：

~~~bash
createdb medicalqa
alembic upgrade head
~~~

如果使用 PowerShell，也可以先临时设置连接串：

~~~powershell
$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5431/medicalqa"
alembic upgrade head
~~~

迁移会创建 5 张业务表：

- <code>users</code> / <code>sessions</code>：认证和登录会话
- <code>conversations</code>：诊断会话和当前状态
- <code>messages</code>：用户/助手消息文本
- <code>qa_turns</code>：每轮关键词、候选池、prompt 实体和记忆状态快照

---

### 6. 启动后端和前端

后端：

~~~bash
python backend/run_dev.py
~~~

默认监听：

~~~text
http://127.0.0.1:8000
~~~

前端：

~~~bash
cd frontend
npm.cmd install
npm.cmd run dev
~~~

默认访问：

~~~text
http://localhost:5173
~~~

注册账号后即可登录使用。管理员账号需要在数据库中将 <code>users.is_admin</code> 置为 <code>true</code>，管理员前端会展示每轮 SSE <code>debug</code> 信息。

---

## 🐳 Docker Compose 启动

现在 Docker Compose 只负责启动基础数据库服务，前端和后端仍按本地开发命令启动：

- <code>postgres</code>：PostgreSQL 业务数据库
- <code>neo4j</code>：Neo4j 知识图谱数据库

首次使用建议先复制并修改环境变量：

~~~bash
cp .env.example .env
~~~

Windows PowerShell：

~~~powershell
Copy-Item .env.example .env
~~~

请至少修改 <code>NEO4J_PASSWORD</code>、<code>SESSION_SECRET</code> 和当前 LLM provider 对应的 API Key。然后启动数据库：

~~~bash
docker compose up -d postgres neo4j
~~~

首次部署时，数据库容器启动后仍然用本地 Python 命令完成初始化：

~~~bash
alembic upgrade head
python build_up_graph.py
python build_vector_index.py
~~~

也就是说，Compose 只帮你省掉 PostgreSQL 和 Neo4j 的安装/启动；图谱导入、向量索引、后端和前端仍按普通本地流程执行：

1. <code>alembic upgrade head</code>：初始化 PostgreSQL 表结构。
2. <code>python build_up_graph.py</code>：把医疗 JSON 数据导入 Neo4j。
3. <code>python build_vector_index.py</code>：构建 FAISS 向量索引。
4. <code>python backend/run_dev.py</code>：启动后端。
5. <code>cd frontend && npm.cmd run dev</code>：启动前端。

默认数据库访问地址：

| 服务 | 地址 |
|---|---|
| Neo4j Browser | <code>http://localhost:7474</code> |
| Neo4j Bolt | <code>bolt://localhost:7687</code> |
| PostgreSQL | <code>localhost:5431</code> |

常用命令：

~~~bash
docker compose ps
docker compose logs -f postgres
docker compose logs -f neo4j
docker compose down
~~~

如果要连同 PostgreSQL / Neo4j 数据卷一起清空：

~~~bash
docker compose down -v
~~~

本机已有端口冲突时，可在 <code>.env</code> 中修改 <code>POSTGRES_PORT</code>、<code>NEO4J_HTTP_PORT</code>、<code>NEO4J_BOLT_PORT</code>。

---

## ✅ 启动前自检

| 检查项 | 验证方式 |
|---|---|
| Neo4j 正在运行 | <code>http://localhost:7474</code> 可打开，或后端能连接 <code>bolt://localhost:7687</code> |
| 知识图谱已导入 | <code>MATCH (n) RETURN count(n)</code> 返回约 4.4 万 |
| 向量索引已构建 | <code>tmp_data/vector_index/entities.faiss</code> 和 <code>entities_meta.npz</code> 存在 |
| PostgreSQL 可连接 | <code>DATABASE_URL</code> 指向的 <code>medicalqa</code> 数据库存在 |
| 数据库迁移已应用 | <code>alembic upgrade head</code> 成功 |
| LLM 可用 | 云厂商 API Key 已设置，或 Ollama 已拉取对应模型 |
| 前端跨域配置正确 | <code>CORS_ORIGINS</code> 包含 <code>http://localhost:5173</code> |

---

## 🧭 系统流程

### 单轮模式

当 <code>CLARIFICATION_ENABLED=False</code>，或调用服务时没有数据库会话，系统走单轮兼容路径：

~~~text
用户问题
  → extract_keywords()
  → Retriever.retrieve()
  → build_prompt()
  → llm_chat_stream()
  → SSE token 输出
~~~

### 多轮诊断模式

默认启用多轮诊断：

~~~text
第 1 轮
用户消息
  → 创建 conversation
  → LLM 提取 local/global keywords
  → FAISS 召回实体 + Neo4j 补充属性/邻居
  → 保存 candidate_pool 和 prompt_entities
  → build_diagnostic_prompt()
  → LLM 生成追问

第 2+ 轮
用户补充症状
  → 加载上一轮 candidate_pool
  → retrieve_followup() 用证据文本重评分
  → detect_topic_change() 判断是否换题
  → decide_action() 规则路由
  → ask：继续追问
  → answer：build_final_prompt() 生成最终回答并追加免责声明
~~~

<code>decide_action()</code> 的路由优先级：

1. <code>turn >= max_turns</code> → <code>answer</code>
2. 候选疾病只剩 1 个且评分 <code>> 0.4</code> → <code>answer</code>
3. 候选疾病为 2 个，分数断层 <code>> 0.15</code> 且 top 分数 <code>> 0.6</code> → <code>answer</code>
4. 已收集 3 个及以上症状且 top 疾病评分 <code>> 0.5</code> → <code>answer</code>
5. 否则继续 <code>ask</code>

---

## 🏗️ 项目结构

~~~text
MedicalQA/
├── backend/
│   ├── main.py                       # FastAPI 应用、CORS、lifespan 预加载
│   ├── run_dev.py                    # 开发启动入口
│   ├── routers/
│   │   ├── auth.py                   # 注册、登录、登出、/me
│   │   └── chat.py                   # 模型列表、SSE 问答、会话 CRUD
│   ├── services/
│   │   ├── qa_service.py             # 核心编排：关键词→检索→路由→Prompt→LLM
│   │   ├── diagnostic.py             # 诊断状态机、路由规则、换题检测
│   │   ├── conversation_service.py   # 会话、消息、QA 轮次 DB CRUD
│   │   └── auth_service.py           # Argon2 密码哈希、Session、登录限流
│   ├── db/
│   │   ├── database.py               # SQLAlchemy engine / session
│   │   └── models.py                 # ORM 模型
│   ├── alembic/                      # 数据库迁移
│   └── schemas.py                    # Pydantic 请求/响应模型
├── frontend/
│   ├── src/
│   │   ├── api.ts                    # API 客户端和 SSE 解析
│   │   ├── App.vue                   # 登录、聊天、历史会话、调试面板
│   │   └── styles.css                # 页面样式
│   └── package.json
├── keyword_extractor.py              # LLM local/global keywords 提取
├── retriever.py                      # 双层检索和后续轮次重评分
├── retrieval_intents.py              # global keywords 到 KG 属性/关系的映射
├── prompt_builder.py                 # 单轮、追问、最终回答 Prompt
├── vector_index.py                   # FAISS 向量索引封装
├── kg_client.py                      # Neo4j 查询封装
├── webui.py                          # LLM 调用层（OpenAI 兼容 / Ollama）
├── config.py                         # .env 加载和 Settings
├── build_up_graph.py                 # 从 JSON 构建 Neo4j 知识图谱
├── build_vector_index.py             # 构建 FAISS 实体向量索引
├── debug_pipeline.py                 # 命令行调试流水线
├── ner/                              # 历史 NER 训练代码，运行时主链路不再依赖
├── data/                             # 医疗原始数据、实体词表、训练数据
├── img/                              # README 图片，暂未随新版完全更新
└── requirements.txt
~~~

---

## 🧠 核心模块

### backend/services/qa_service.py

系统核心编排层。它根据配置和 <code>conversation_id</code> 自动选择三条路径：

- <code>_stream_single_turn()</code>：单轮问答兼容模式。
- <code>_stream_first_turn()</code>：创建新诊断会话，建立候选池并生成首个追问。
- <code>_stream_followup_turn()</code>：加载历史候选池，重评分、路由，并继续追问或最终回答。

所有路径都通过 SSE 输出 <code>status</code>、<code>debug</code>、<code>token</code>、<code>action</code>、<code>done</code>、<code>error</code> 事件。

### retriever.py

检索层采用 LightRAG 风格的双层思路：

- low-level：<code>local_keywords</code> → FAISS 语义召回实体。
- high-level：<code>global_keywords</code> → 按 <code>retrieval_intents.py</code> 映射补充疾病属性和关系。
- follow-up：对上一轮候选池构造证据文本，用新症状上下文重评分，并补充新增实体。

后续轮次综合排名使用旧分数、上下文相似度和症状匹配加分：

~~~text
new_rank = 0.5 * old_rank + 0.3 * ctx_score + 0.2 * symptom_bonus
~~~

### prompt_builder.py

Prompt 分为三类：

- <code>build_prompt()</code>：单轮模式，直接基于知识图谱信息回答。
- <code>build_diagnostic_prompt()</code>：追问模式，展示候选疾病、症状、治疗方向、药品等证据，引导 LLM 继续问关键问题。
- <code>build_final_prompt()</code>：最终回答模式，输出可能疾病、依据、关联药品和治疗方向，并写入用药安全约束。

### backend/services/diagnostic.py

负责诊断状态结构和规则路由：

- <code>DiagnosticState</code>：保存轮次、阶段、症状、历史实体、topic_id。
- <code>CandidateEntity</code>：保存候选实体评分、来源、属性和邻居关系。
- <code>detect_topic_change()</code>：检测用户是否在会话中切换到新主题。
- <code>decide_action()</code>：不调用 LLM，使用稳定规则决定追问或回答。

### webui.py

虽然文件名仍叫 <code>webui.py</code>，当前实际职责已经是 LLM 调用层：

- <code>llm_generate()</code>：非流式文本生成，用于关键词提取。
- <code>llm_chat_stream()</code>：流式聊天生成，用于 SSE token 输出。
- <code>LLM_PROVIDER=ollama</code> 时调用本地 Ollama；其他 provider 走 OpenAI 兼容接口。

---

## 🧬 知识图谱

<p align="center">
  <img src="img/graph.png" width="75%" />
  <br>
  <sub>知识图谱示例图暂沿用旧版截图。</sub>
</p>

实体类型：

| 类型 | 中文含义 | 规模 |
|---|---|---:|
| 疾病 | 疾病实体 | 8,808 |
| 药品 | 药品实体 | 3,828 |
| 食物 | 宜吃/忌吃食物 | 4,870 |
| 检查项目 | 疾病相关检查 | 3,353 |
| 科目 | 就诊科室 | 54 |
| 药品商 | 药品生产商 | 17,201 |
| 疾病症状 | 症状实体 | 5,998 |
| 治疗方法 | 治疗方式 | 544 |

疾病属性包括：疾病简介、疾病病因、预防措施、治疗周期、治愈概率、疾病易感人群、名称。

主要关系包括：疾病使用药品、疾病宜吃食物、疾病忌吃食物、疾病所需检查、疾病所属科目、疾病的症状、治疗的方法、疾病并发疾病、药品商生产药品等。

---

## 💻 前端界面

<p align="center">
  <img src="img/login.png" width="55%" />
  <br>
  <sub>登录界面截图暂沿用旧版图片。</sub>
</p>

<p align="center">
  <img src="img/e1.png" width="80%" />
  <br>
  <sub>问答界面截图暂沿用旧版图片。</sub>
</p>

当前 Vue3 前端提供：

- 登录、注册、登出和当前用户识别。
- 左侧历史会话列表，支持打开和删除会话。
- 模型下拉选择，来自后端 <code>/api/models</code>。
- <code>final_entity_top_k</code> 控制送入 prompt 的实体数量。
- SSE 流式显示回答 token 和状态提示。
- Markdown 渲染助手回答。
- 管理员展开查看每轮 <code>debug</code> 事件内容。

---

## ⚙️ 配置说明

所有配置集中在 <code>.env.example</code> 和 <code>config.py</code>，启动时 <code>config.py</code> 会自动加载项目根目录下的 <code>.env</code>。

| 变量名 | 默认/示例 | 说明 |
|---|---|---|
| <code>NEO4J_URL</code> | <code>bolt://localhost:7687</code> | Neo4j Bolt 地址 |
| <code>NEO4J_USER</code> | <code>neo4j</code> | Neo4j 用户名 |
| <code>NEO4J_PASSWORD</code> | 本地填写 | Neo4j 密码 |
| <code>NEO4J_DBNAME</code> | <code>neo4j</code> | Neo4j 数据库名 |
| <code>LLM_PROVIDER</code> | <code>dashscope</code> | <code>dashscope</code> / <code>mimo</code> / <code>deepseek</code> / <code>ollama</code> |
| <code>DASHSCOPE_API_KEY</code> | 本地填写 | 阿里百炼 API Key |
| <code>MIMO_API_KEY</code> | 本地填写 | 小米 MiMo API Key |
| <code>DEEPSEEK_API_KEY</code> | 本地填写 | DeepSeek API Key |
| <code>LLM_API_KEY</code> | 可选 | 通用覆盖，优先级高于厂商专属 Key |
| <code>LLM_BASE_URL</code> | 可选 | 自定义 OpenAI 兼容接口地址 |
| <code>LLM_MODEL</code> | 可选 | 自定义默认模型 |
| <code>LLM_MODELS</code> | 可选 | 前端可选模型列表，逗号分隔 |
| <code>OLLAMA_QWEN_MODEL</code> | <code>qwen:32b</code> | Ollama Qwen 模型标签 |
| <code>OLLAMA_LLAMA_MODEL</code> | <code>llama2-chinese:13b-chat-q8_0</code> | Ollama Llama 模型标签 |
| <code>DATABASE_URL</code> | <code>postgresql+psycopg://.../medicalqa</code> | PostgreSQL 连接串 |
| <code>SESSION_SECRET</code> | <code>dev-insecure-change-me</code> | Session 服务端密钥，生产必须修改 |
| <code>SESSION_COOKIE_NAME</code> | <code>session</code> | 登录 Cookie 名称 |
| <code>SESSION_EXPIRE_DAYS</code> | <code>7</code> | Session 有效期 |
| <code>COOKIE_SECURE</code> | <code>False</code> | HTTPS 生产环境建议 <code>True</code> |
| <code>CORS_ORIGINS</code> | <code>http://localhost:5173</code> | 允许携带 Cookie 访问 API 的前端源 |
| <code>DATA_DIR</code> | <code>data</code> | 数据目录 |
| <code>TMP_DIR</code> | <code>tmp_data</code> | 临时和索引目录 |
| <code>MODEL_DIR</code> | <code>model</code> | 本地模型目录 |
| <code>MAX_CLARIFICATION_TURNS</code> | <code>2</code> | 多轮诊断最大追问轮次 |
| <code>CLARIFICATION_ENABLED</code> | <code>True</code> | 是否启用多轮诊断 |
| <code>LOG_LEVEL</code> | <code>INFO</code> | 日志级别 |

---

## 🔌 后端 API

所有业务 API 前缀均为 <code>/api</code>。除注册、登录外，聊天和会话接口都需要登录 Cookie。

| 方法 | 路径 | 说明 |
|---|---|---|
| <code>POST</code> | <code>/api/auth/register</code> | 注册普通用户，用户名 3-64 字符，密码 8-128 字符 |
| <code>POST</code> | <code>/api/auth/login</code> | 登录并设置 HttpOnly Session Cookie |
| <code>POST</code> | <code>/api/auth/logout</code> | 登出并撤销 Session |
| <code>GET</code> | <code>/api/auth/me</code> | 获取当前登录用户 |
| <code>GET</code> | <code>/api/models</code> | 返回当前 provider 和可选模型 |
| <code>POST</code> | <code>/api/chat/stream</code> | SSE 流式问答 |
| <code>GET</code> | <code>/api/conversations</code> | 当前用户会话列表 |
| <code>GET</code> | <code>/api/conversations/{id}/messages</code> | 某会话消息历史 |
| <code>DELETE</code> | <code>/api/conversations/{id}</code> | 删除会话及关联消息、QA 轮次 |

<code>POST /api/chat/stream</code> 请求体：

~~~json
{
  "message": "我牙齿上长黑斑怎么回事",
  "model": "deepseek-v4-flash",
  "final_entity_top_k": 5,
  "conversation_id": null
}
~~~

SSE 事件类型：

| 事件 | 数据 | 说明 |
|---|---|---|
| <code>status</code> | string | 当前进度，如“正在提取关键词...” |
| <code>debug</code> | object | 关键词、检索结果、prompt 等调试信息 |
| <code>token</code> | string | LLM 输出片段 |
| <code>action</code> | object | 本轮路由结果，<code>type</code> 为 <code>ask</code> 或 <code>answer</code> |
| <code>done</code> | object | 完成事件，包含 <code>conversation_id</code> 和 <code>diagnostic_state</code> |
| <code>error</code> | <code>{ "message": string }</code> | 错误信息 |

---

## 🧪 调试

命令行快速跑一遍检索和生成流程：

~~~bash
python debug_pipeline.py "我牙齿上长黑斑怎么回事"
~~~

后端日志等级可通过 <code>.env</code> 设置：

~~~env
LOG_LEVEL=DEBUG
~~~

前端管理员用户可以在聊天消息下展开查看后端推送的 <code>debug</code> 事件，包括关键词、候选实体、prompt 和知识图谱信息。

---

## ⚠️ 医疗与安全说明

- 本项目仅用于医疗知识问答和 RAG 系统研究，不构成真实医疗诊断或治疗建议。
- Prompt 中明确禁止输出剂量、疗程、用法用量，禁止说“你可以吃 X”。
- 最终回答由后端强制追加免责声明，提醒用户就医并避免自行用药。
- 生产环境必须修改 <code>SESSION_SECRET</code>，开启 HTTPS 后将 <code>COOKIE_SECURE=True</code>。
- 不要提交 <code>.env</code>、API Key、数据库密码、Neo4j 密码或任何用户隐私数据。

---

## 📌 当前状态与注意事项

- 当前主线是 FastAPI + Vue3，不是旧版 Streamlit UI。
- <code>webui.py</code> 名称保留，但实际负责 LLM 调用，不再是主 UI。
- <code>ner/</code> 保留训练与历史实验代码，运行时检索主链路使用 LLM 关键词提取 + FAISS。
- <code>requirements.txt</code> 当前没有显式锁定 <code>faiss-cpu</code> 和 <code>sentence-transformers</code>，首次构建向量索引前请按上文补装。
- README 中图片暂未同步新版界面，后续可以单独更新截图和架构图。

---

## 🙏 致谢

- 数据集：[Open-KG 疾病信息数据集](http://data.openkg.cn/dataset/disease-information)
- 参考：[RAGOnMedicalKG](https://github.com/liuhuanyong/RAGOnMedicalKG)
- 参考：[QASystemOnMedicalKG](https://github.com/liuhuanyong/QASystemOnMedicalKG)
- 思路参考：[LightRAG](https://github.com/HKUDS/LightRAG)
