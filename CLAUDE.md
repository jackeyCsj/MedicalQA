# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

基于知识图谱 RAG 的医疗智能问答系统，结合 Neo4j 知识图谱（4.4 万实体、31 万关系）、BERT NER 和大语言模型。流水线：用户提问 → NER 实体抽取 → LLM 意图识别 → Cypher 图谱查询 → LLM 答案生成。

## 常用命令

```bash
# 启动 Streamlit 应用
streamlit run login.py

# 构建知识图谱（注意：会询问是否删除 Neo4j 中已有数据）
python build_up_graph.py --website http://localhost:7474 --user neo4j --password <密码> --dbname neo4j

# 生成 NER 训练数据（需要先构建知识图谱生成 ent_aug/ 文件）
python ner_data.py

# 训练 NER 模型
python -m ner.train

# 运行测试
pytest tests/
```

## 架构设计

**4 层架构，config.py 横切配置注入：**

```
login.py（Streamlit 入口）→ webui.py（对话编排）→ intent_router.py（表驱动路由）→ kg_client.py（Neo4j 查询）
                                    ↓
                              ner/（NER 子包）
                                    ↓
                              config.py（集中配置）
```

**数据流：**
1. `webui.py:load_model()` 加载 BERT+RNN NER 模型（被 Streamlit 缓存）
2. `webui.py:Intent_Recognition()` 调用 ollama LLM 将用户意图分为 16 类
3. `intent_router.py:route_intents()` 解析 LLM 输出，按关键字长度降序匹配避免子串冲突（如"治疗周期"先于"治疗方法"）
4. `kg_client.py:KGClient` 执行 Cypher 查询，返回数据或空列表/None
5. `webui.py:main()` 通过 `ollama.chat(stream=True)` 流式生成 LLM 回答

**关键设计决策：**
- 表驱动意图路由（`intent_router.py` 中的 `INTENT_SPECS`）替代 16 个 if-else 块
- NER 采用双重策略：BERT 模型预测 + Aho-Corasick 规则匹配，按实体长度优先级合并
- TF-IDF 对齐将抽取的实体映射到知识图谱标准实体名（相似度阈值 ≥ 0.5）
- 所有外部配置通过 `config.py` 中的环境变量管理（默认值与原代码一致）

## 配置说明

环境变量（默认值见 `config.py`）：

| 变量名 | 默认值 | 用途 |
|--------|--------|------|
| `NEO4J_URL` | `bolt://localhost:7687` | Neo4j 连接地址 |
| `NEO4J_PASSWORD` | *（请自行设置）* | Neo4j 密码 |
| `OLLAMA_QWEN_MODEL` | `qwen:32b` | 主力 LLM 模型 |
| `NER_MODEL_NAME` | `model/chinese-roberta-wwm-ext` | BERT 基座路径 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

建议在项目根目录创建 `.env` 文件（已加入 `.gitignore`），启动前执行：`export $(cat .env | xargs)`

## NER 模块（`ner/`）

- **model.py**：`Bert_Model` 类 — BERT → 双向 RNN（属性名 `self.gru` 是历史遗留命名，实际是 `nn.RNN`）→ 线性分类器 → BIO 标签
- **dataset.py**：`Entity_Extend` 类，3 种数据增强策略（实体替换/掩码/拼接）；增强仅在 epoch ≥ 7 的奇数轮触发
- **inference.py**：`get_ner_result()` — 模型预测 + 规则匹配 + 合并 + TF-IDF 对齐
- **train.py**：训练入口；保存 `model/best_roberta_rnn_model_ent_aug.pt` 和 `tmp_data/tag2idx.npy`

## 知识图谱

8 类实体：疾病、药品、食物、检查项目、科目、药品商、疾病症状、治疗方法

11 类关系：疾病使用药品、疾病宜吃食物、疾病忌吃食物、疾病所需检查、疾病所属科目、疾病的症状、治疗的方法、疾病并发疾病 等



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
