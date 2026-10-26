# -*- coding: utf-8 -*-
"""电商客服 RAG Agent 的 Web 服务（FastAPI）

把 agent.py 的对话能力包装成 HTTP 接口 + 网页前端，
让项目从「本地脚本」变成「能访问的完整系统」。

运行：python server.py   （默认 http://127.0.0.1:8000）
"""
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agent  # 复用 agent.py 的知识库 + 检索 + 对话

app = FastAPI(title="电商客服 RAG Agent")

BASE = os.path.dirname(os.path.abspath(__file__))


class ChatReq(BaseModel):
    message: str
    history: list = []


@app.post("/api/chat")
def chat(req: ChatReq):
    """一轮对话：返回 Agent 回答。"""
    answer = agent.chat(req.message, req.history)
    return {"answer": answer}


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(agent.CHUNKS)}


app.mount("/", StaticFiles(directory=os.path.join(BASE, "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
