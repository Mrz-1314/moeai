"""
Simple FastAPI application that exposes a unified chat endpoint for
multiple AI providers (OpenAI GPT‑4o, Google Gemini and xAI Grok).

The routing logic is simple: the client specifies which provider to
use via the `provider` field. This is a proof‑of‑concept implementation
and does not include advanced MoE routing – you can extend the logic
in the `route_request` function.

Environment variables expected (see README for details):

    OPENAI_API_KEY    – API key for OpenAI (ChatGPT / GPT‑4o).
    GEMINI_API_KEY    – API key for Google Gemini.
    XAI_API_KEY       – API key for xAI Grok.

Run this service with uvicorn:

    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in your browser to access the basic
chat UI.
"""

import os
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Third‑party clients
# Optional imports: these modules may not be installed in all environments.
try:
    import openai  # type: ignore
except Exception:
    openai = None  # type: ignore

import requests  # requests is part of the base environment

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore

# Load environment variables from .env if present
load_dotenv()

# Configure OpenAI and Gemini clients
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
XAI_API_KEY = os.getenv("XAI_API_KEY")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    provider: str
    messages: list[ChatMessage]


app = FastAPI(title="Unified AI Chat Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For simplicity; adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def call_openai(messages: list[ChatMessage]) -> str:
    """Call the OpenAI Chat Completion API with GPT‑4o and return the first message."""
    # If the openai package is not available, return a placeholder message
    if openai is None:
        return "OpenAI SDK is not installed in this environment."
    if not openai.api_key:
        # If no API key is configured, return a helpful placeholder
        return "OpenAI API key is not configured on the server."
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[m.dict() for m in messages],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def call_gemini(messages: list[ChatMessage]) -> str:
    """Call the Gemini API and return the generated content."""
    # If the generativeai SDK is not available or no API key is configured, return placeholder
    if genai is None:
        return "Google Gemini SDK is not installed in this environment."
    if not os.getenv("GEMINI_API_KEY"):
        return "Gemini API key is not configured on the server."
    try:
        # The Gemini SDK expects a flat list of contents. We'll join user
        # and assistant messages into a single prompt separated by newlines.
        # Note: This is a simplified example; for more complex contexts
        # you may want to structure the prompt differently.
        prompt = "\n".join(m.content for m in messages)
        client = genai.Client()
        # Use the preview model; adjust as needed
        result = client.models.generate_content(
            model="gemini-2.5-pro-preview-06-05",
            contents=prompt
        )
        return result.text
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def call_xai(messages: list[ChatMessage]) -> str:
    """Call the xAI Grok API using a simple HTTP request."""
    if not XAI_API_KEY:
        # If no key configured, return placeholder text
        return "xAI API key is not configured on the server."
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "grok-3-beta",
        "messages": [m.dict() for m in messages],
        "stream": False,
        "temperature": 0.7,
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        resp_json = resp.json()
        return resp_json["choices"][0]["message"]["content"]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def route_request(provider: str, messages: list[ChatMessage]) -> str:
    """
    Route the request to the appropriate provider based on the `provider` string.

    Valid providers: "openai", "gemini", "xai".

    Raises HTTPException for unsupported provider.
    """
    provider = provider.lower().strip()
    if provider == "openai" or provider == "chatgpt":
        return call_openai(messages)
    elif provider == "gemini" or provider == "google":
        return call_gemini(messages)
    elif provider == "xai" or provider == "grok":
        return call_xai(messages)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the basic chat user interface."""
    root = os.path.dirname(os.path.realpath(__file__))
    html_path = os.path.join(root, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content)


@app.post("/chat")
async def chat_endpoint(req: ChatRequest) -> JSONResponse:
    """Chat endpoint that dispatches to the appropriate provider."""
    # Validate messages list is non-empty
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")
    # Call provider
    answer = route_request(req.provider, req.messages)
    return JSONResponse({"content": answer})