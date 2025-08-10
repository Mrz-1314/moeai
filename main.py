# main.py
# FastAPI 后端：统一接入 OpenAI(ChatGPT)、Google Gemini、xAI Grok
# - /         ：返回 index.html
# - /chat     ：{ provider: "openai"|"gemini"|"xai"|"auto", messages: [...] }
# 说明：
# 1) 不强制安装各家 SDK，全部用 requests 直连官方 REST API，Vercel/Docker 更稳。
# 2) 环境变量：OPENAI_API_KEY / GEMINI_API_KEY / XAI_API_KEY
# 3) 简单 MoE 路由：provider=auto 时按启发式分发；你可自行完善 route_request()

from __future__ import annotations
import os
import re
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ------------ 配置 ------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()

# 默认模型（可在 /chat 请求里覆盖）
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-preview-06-05")
DEFAULT_XAI_MODEL = os.getenv("XAI_MODEL", "grok-3")

# ------------ FastAPI 基础 ------------
app = FastAPI(title="Unified AI Chat (MoE Router)")

# CORS（前后端分离时把 "*" 换成你的域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态/首页（Docker 部署时可直接用）
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
def root():
    # 如果你的 index.html 在根目录，这里会正常返回
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse({"ok": True, "msg": "index.html not found in project root"})


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "openai": bool(OPENAI_API_KEY),
        "gemini": bool(GEMINI_API_KEY),
        "xai": bool(XAI_API_KEY),
    }


# ------------ 数据模型 ------------
class ChatMessage(BaseModel):
    role: str = Field(..., description="user/assistant/system")
    content: str


class ChatRequest(BaseModel):
    provider: str = Field(..., description='"openai"|"gemini"|"xai"|"auto"')
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1500


# ------------ 路由器（简易 MoE） ------------
TIME_WORDS = ("今天", "刚刚", "最新", "实时", "news", "today", "now", "current", "breaking")
MEDIA_WORDS = ("图片", "截图", "视频", "图像", "image", "photo", "video", "vision")
CODE_WORDS = ("def ", "class ", "console.log", "Exception", "Traceback", "SELECT ", "INSERT ", "{", "};")

def route_request(messages: List[ChatMessage]) -> str:
    """非常简单的启发式路由：可按你业务继续加强"""
    text = " \n".join(m.content for m in messages[-3:]).lower()
    # 含媒体/多模态→Gemini
    if any(w.lower() in text for w in MEDIA_WORDS):
        return "gemini"
    # 强时效→Grok
    if any(w.lower() in text for w in TIME_WORDS):
        return "xai"
    # 代码/结构化→OpenAI
    if any(w.lower() in text for w in CODE_WORDS):
        return "openai"
    # 默认：OpenAI
    return "openai"


# ------------ Provider 适配：HTTP 直连 ------------
def call_openai(messages: List[ChatMessage], model: Optional[str], temperature: float, max_tokens: int) -> str:
    if not OPENAI_API_KEY:
        return "OpenAI API key 未配置（OPENAI_API_KEY）。"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model or DEFAULT_OPENAI_MODEL,
        "messages": [m.dict() for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return str(data)


def _gemini_convert_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """
    将 OpenAI 风格 messages 转为 Gemini REST 的 contents：
    - Gemini 角色：user / model
    - system 信息：拼到第一条 user 前面
    """
    system_prompts = [m.content for m in messages if m.role == "system"]
    sys_text = ("\n".join(system_prompts)).strip()
    contents = []
    for m in messages:
        if m.role == "system":
            continue
        role = "user" if m.role == "user" else "model"
        text = m.content
        if role == "user" and sys_text:
            text = f"[System]\n{sys_text}\n\n[User]\n{text}"
            sys_text = ""  # 只拼一次
        contents.append({"role": role, "parts": [{"text": text}]})
    if not contents and sys_text:
        contents.append({"role": "user", "parts": [{"text": f"[System]\n{sys_text}"}]})
    return contents


def call_gemini(messages: List[ChatMessage], model: Optional[str], temperature: float, max_tokens: int) -> str:
    if not GEMINI_API_KEY:
        return "Gemini API key 未配置（GEMINI_API_KEY）。"
    use_model = model or DEFAULT_GEMINI_MODEL
    # Gemini REST：v1beta
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{use_model}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": _gemini_convert_messages(messages),
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    resp = requests.post(url, json=body, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    # 解析 candidates → parts → text
    try:
        parts = data["candidates"][0]["content"]["parts"]
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "\n".join(t for t in texts if t)
    except Exception:
        return str(data)


def call_xai(messages: List[ChatMessage], model: Optional[str], temperature: float, max_tokens: int) -> str:
    if not XAI_API_KEY:
        return "xAI API key 未配置（XAI_API_KEY）。"
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model or DEFAULT_XAI_MODEL,
        "messages": [m.dict() for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return str(data)


# ------------ 统一入口 ------------
@app.post("/chat")
def chat(req: ChatRequest):
    """
    请求示例：
    {
      "provider": "auto",
      "messages": [
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"帮我写个SQL分组示例"}
      ],
      "temperature": 0.7,
      "max_tokens": 800
    }
    """
    provider = req.provider.lower().strip()
    temperature = float(req.temperature or 0.7)
    max_tokens = int(req.max_tokens or 1500)

    if provider not in {"openai", "gemini", "xai", "auto"}:
        raise HTTPException(status_code=400, detail="provider 必须是 openai | gemini | xai | auto")

    # 自动路由
    if provider == "auto":
        provider = route_request(req.messages)

    try:
        if provider == "openai":
            content = call_openai(req.messages, req.model, temperature, max_tokens)
        elif provider == "gemini":
            content = call_gemini(req.messages, req.model, temperature, max_tokens)
        elif provider == "xai":
            content = call_xai(req.messages, req.model, temperature, max_tokens)
        else:
            raise HTTPException(status_code=400, detail="未知 provider")
        return {"provider": provider, "content": content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")


# 本地调试：python main.py
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "3000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

        raise HTTPException(status_code=400, detail="Messages list cannot be empty")
    # Call provider
    answer = route_request(req.provider, req.messages)
    return JSONResponse({"content": answer})
