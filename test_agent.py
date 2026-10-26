# -*- coding: utf-8 -*-
import agent

history = []
tests = [
    "我买的衣服尺码不合适能退吗",
    "满99元包邮，不满99元运费多少钱",
    "你好，在吗",
    "XQ-2000 空气净化器适用多大面积",
]
for q in tests:
    print("=" * 60)
    print("🙋 你：", q)
    ans = agent.chat(q, history)
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": ans})
    print("🤖 客服：", ans)
    print()

# 多轮：追问，看 Agent 是否记得上下文
print("=" * 60)
print("🙋 你：那退货的运费算谁的？")
ans = agent.chat("那退货的运费算谁的？", history)
print("🤖 客服：", ans)
