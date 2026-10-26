# 电商客服 RAG Agent

把「RAG 检索」这个单点能力，串成一个**完整业务流程的客服 Agent**。

这是从 `rag-eval`（检索评测）演进而来的应用层项目：评测先量化了检索策略的适用边界，这里再把检索能力封装成工具、交给 Agent 自主调度。

## 架构

```
用户提问
  ↓
Agent（Function Calling）
  ├─ 需要查资料？→ 调用 search_knowledge 工具
  │      └─ 混合检索：BM25 + bge-m3 向量 → RRF 融合 → top-k 片段
  ├─ 闲聊/寒暄？→ 直接回答（不检索）
  ↓
基于检索结果生成回答（带引用来源）
  ↓
多轮对话（保持上下文）
  ↓
检索不到 → 明确说不知道，绝不编造
```

## 核心特性

- **Agentic RAG**：检索是 Agent 的「工具」，由 Agent 自主决定何时检索，而不是每次机械地先检索再回答。闲聊不检索、知识问题才检索。
- **混合检索**：BM25（精确关键词）+ bge-m3 向量（语义），RRF 融合，覆盖「满99包邮」这类精确问题和「衣服不合适能退吗」这类口语化问题。
- **多轮对话**：维护对话历史，支持追问（「那运费算谁的」能承接上文的退货语境）。
- **引用溯源**：每个回答标注依据的文档片段，可追溯、可核查。
- **防编造**：资料里没有的信息明确说「建议联系人工客服」，不幻觉。

## 快速开始

前置：本地 Ollama 运行 `bge-m3`，DeepSeek API Key 在 `E:\agent-learning\.env`。

```bash
pip install -r requirements.txt

# Web 服务（推荐）：启动后浏览器打开 http://127.0.0.1:8000
python server.py

# 命令行多轮对话
python agent.py

# 自动跑一组测试问题
python test_agent.py
```

### 接口

- `GET /health` — 健康检查
- `POST /api/chat` — 对话接口，请求 `{"message": "问题", "history": [{"role":"user","content":"..."}]}`，返回 `{"answer": "..."}`

## 运行效果

```
🙋 你：我买的衣服尺码不合适能退吗
  🔍 Agent 检索：衣服尺码不合适退货政策
🤖 客服：可以的。7 天内不影响二次销售可申请无理由退货...
（来源：退换货政策::七天无理由退货、换货规则）

🙋 你：你好，在吗
🤖 客服：您好，我在的～请问有什么可以帮您？（不检索，直接答）

🙋 你：那退货的运费算谁的？   ← 多轮追问
  🔍 Agent 检索：退货运费由谁承担
🤖 客服：七天无理由退货运费买家承担；质量问题卖家承担...
（来源：退换货政策::无理由退货运费承担）
```

## 与 rag-eval 的关系

| 项目 | 定位 |
|---|---|
| [rag-eval](https://github.com/Whybeyou/rag-eval) | 检索策略评测：量化 BM25 / 向量 / 混合的适用边界 |
| ecom-agent（本项目） | 应用层：把评测过的检索能力串成完整 Agent |

先评测、再应用，避免「拍脑袋选检索策略」。

## 技术栈

- Python 3（标准库 + numpy）
- DeepSeek API（Function Calling，Agent 自主决策）
- 本地 Ollama（bge-m3 embedding）
- 手写 BM25 + RRF 融合
