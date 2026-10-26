# -*- coding: utf-8 -*-
"""电商客服 RAG Agent（完整业务流程）

把「RAG 检索」这个单点能力，串成一个完整的客服 Agent：
  1. 检索封装成 search_knowledge 工具（BM25 + bge-m3 混合检索）
  2. Agent 用 Function Calling 自主决定「何时该检索、何时直接回答」
  3. 完整业务流程：提问 → 检索 → 生成(带引用) → 多轮对话 → 兜底不编造

运行：python agent.py（交互式多轮对话）
"""
import json
import math
import os
import re
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE, "knowledge")

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-pro"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_EMBED_MODEL = "bge-m3"


def load_env():
    path = os.path.join(BASE, "..", ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip())


load_env()
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


# ---------- 基础 ----------
def http_post(url, payload, headers=None, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers=headers or {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def call_llm(messages, tools=None):
    payload = {"model": DEEPSEEK_MODEL, "messages": messages}
    if tools:
        payload["tools"] = tools
    return http_post(DEEPSEEK_URL, payload,
                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})


def embed(text):
    return http_post(OLLAMA_EMBED_URL, {"model": OLLAMA_EMBED_MODEL, "input": text})["embeddings"][0]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = (sum(x * x for x in a)) ** 0.5
    nb = (sum(x * x for x in b)) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def tokenize(text):
    text = re.sub(r"[\s，。！？、；：（）【】「」\"'.,!?;:()\[\]<>]+", "", text)
    return [text[i:i + 2] for i in range(len(text) - 1)]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N if self.N else 0
        self.k1, self.b = k1, b
        self.df = {}
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1

    def _idf(self, t):
        n = self.df.get(t, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1)

    def search(self, q, top_k):
        qt = tokenize(q)
        scored = []
        for i, d in enumerate(self.docs):
            tf = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in qt:
                if t in tf:
                    s += self._idf(t) * (tf[t] * (self.k1 + 1)) / \
                        (tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            scored.append((i, s))
        scored.sort(key=lambda x: -x[1])
        return [i for i, _ in scored[:top_k]]


def rrf_fusion(a, b, k=60):
    score = {}
    for rank, i in enumerate(a):
        score[i] = score.get(i, 0) + 1 / (k + rank + 1)
    for rank, i in enumerate(b):
        score[i] = score.get(i, 0) + 1 / (k + rank + 1)
    return [i for i, _ in sorted(score.items(), key=lambda x: -x[1])]


# ---------- 知识库 + 混合检索 ----------
CHUNKS = []
for fn in sorted(os.listdir(KB_DIR)):
    if fn.endswith(".md"):
        docname = fn[:-3]
        content = open(os.path.join(KB_DIR, fn), encoding="utf-8").read()
        for sec in re.split(r"\n## ", content)[1:]:
            title = sec.split("\n", 1)[0].strip()
            CHUNKS.append({"id": f"{docname}::{title}", "text": "## " + sec.strip()})

TEXTS = [c["text"] for c in CHUNKS]
_VEC = [embed(t) for t in TEXTS]
_BM25 = BM25(TEXTS)


def search_knowledge(query, top_k=4):
    """混合检索：BM25 + 向量 RRF 融合，返回 top-k 片段。"""
    b = _BM25.search(query, top_k * 2)
    qv = embed(query)
    v = sorted(range(len(_VEC)), key=lambda i: cosine(_VEC[i], qv), reverse=True)[:top_k * 2]
    fused = rrf_fusion(b, v)[:top_k]
    return [{"source": CHUNKS[i]["id"], "text": CHUNKS[i]["text"]} for i in fused]


# ---------- 工具（给 Agent 的检索能力） ----------
TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": "检索电商客服知识库，获取与用户问题相关的政策/规则/商品信息片段。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词或改写后的问题"}
            },
            "required": ["query"],
        },
    },
}]

SYSTEM_PROMPT = (
    "你是电商平台的人工客服助手。回答用户关于退换货、物流、支付、发票、售后、商品等的问题。\n"
    "规则：\n"
    "1. 当用户问题需要依据平台政策或规则回答时，先调用 search_knowledge 检索相关资料，再基于资料作答。\n"
    "2. 闲聊、寒暄或不需要查资料的问题，直接回答，不必检索。\n"
    "3. 回答末尾用「（来源：xxx）」标注依据的文档。\n"
    "4. 资料里没有的信息，明确说「这个问题资料里没有，建议联系人工客服」，绝不编造。"
)


def chat(user_msg, history):
    """一轮对话：Agent 自主决定是否检索，返回回答文本。history 为 [{role, content}, ...]"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_msg}]
    for _ in range(4):
        resp = call_llm(messages, tools=TOOLS)
        msg = resp["choices"][0]["message"]
        if not msg.get("tool_calls"):
            return msg["content"]
        messages.append(msg)
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            print(f"  🔍 Agent 检索：{args['query']}")
            hits = search_knowledge(args["query"])
            result = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    return "（检索轮次超限）"


def main():
    if not API_KEY:
        raise SystemExit("请先设置 DEEPSEEK_API_KEY（E:\\agent-learning\\.env）")
    print(f"电商客服 Agent 已就绪（知识库 {len(CHUNKS)} 段）。输入问题回车，输入 quit 退出。\n")
    history = []
    while True:
        q = input("🙋 你：").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit", "退出"):
            break
        ans = chat(q, history)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": ans})
        print(f"🤖 客服：{ans}\n")


if __name__ == "__main__":
    main()
