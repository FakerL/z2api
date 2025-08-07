"""
Proxy handler for Z.AI API requests – fixed first-char loss after </think>
"""

import json
import logging
import re
import time
import uuid
from typing import AsyncGenerator, Dict, Any, Tuple

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from cookie_manager import cookie_manager
from models import ChatCompletionRequest, ChatCompletionResponse

logger = logging.getLogger(__name__)


class ProxyHandler:
    # -------------------- 初始化 --------------------
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=300.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            http2=True,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    # -------------------- 文字處理 --------------------
    def _balance_think_tag(self, s: str) -> str:
        o, c = len(re.findall(r"<think>", s)), len(re.findall(r"</think>", s))
        if o > c:
            s += "</think>" * (o - c)
        elif c > o:
            for _ in range(c - o):
                s = re.sub(r"</think>(?!.*</think>)", "", s, count=1)
        return s

    def _clean_thinking_content(self, s: str) -> str:
        if not s:
            return s
        s = re.sub(r'<details[^>]*>', '', s)
        s = re.sub(r'</details>', '', s)
        s = re.sub(r'<summary[^>]*>.*?</summary>', '', s, flags=re.DOTALL)
        s = re.sub(r'<[^>]+>', '', s)
        s = re.sub(r'^>\s*', '', s)
        s = re.sub(r'\n>\s*', '\n', s)
        return s

    def _clean_answer_content(self, s: str) -> str:
        if not s:
            return s
        return re.sub(r"<details[^>]*>.*?</details>", "", s, flags=re.DOTALL)

    def _serialize_messages(self, msgs) -> list:
        out = []
        for m in msgs:
            if hasattr(m, "dict"):
                out.append(m.dict())
            elif hasattr(m, "model_dump"):
                out.append(m.model_dump())
            elif isinstance(m, dict):
                out.append(m)
            else:
                out.append({"role": getattr(m, "role", "user"), "content": getattr(m, "content", str(m))})
        return out

    # -------------------- 上游準備 --------------------
    async def _prepare_upstream(self, req: ChatCompletionRequest) -> Tuple[Dict[str, Any], Dict[str, str], str]:
        cookie = await cookie_manager.get_next_cookie()
        if not cookie:
            raise HTTPException(503, "No available cookies")

        target_model = settings.UPSTREAM_MODEL if req.model == settings.MODEL_NAME else req.model
        body = {
            "stream": True,
            "model": target_model,
            "messages": self._serialize_messages(req.messages),
            "background_tasks": {"title_generation": True, "tags_generation": True},
            "chat_id": str(uuid.uuid4()),
            "features": {"enable_thinking": True},
            "id": str(uuid.uuid4()),
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cookie}",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/event-stream",
            "Origin": "https://chat.z.ai",
            "Referer": "https://chat.z.ai/",
        }
        return body, headers, cookie

    # -------------------- 串流處理 --------------------
    async def stream_proxy_response(self, req: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        client = None
        try:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, read=300.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                http2=True,
            )
            body, headers, cookie = await self._prepare_upstream(req)
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            think_open = False
            current_phase = None

            async with client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(cookie)
                    yield self._error_sse(completion_id, req.model, f"Upstream {resp.status_code}")
                    return

                async for raw in resp.aiter_text():
                    for line in filter(None, (x.strip() for x in raw.split("\n"))):
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            if think_open and settings.SHOW_THINK_TAGS:
                                # 若仍開啟，補閉標籤
                                yield self._sse_chunk(completion_id, req.model, "</think>")
                            yield self._done_sse(completion_id, req.model)
                            return

                        try:
                            parsed = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        data = parsed.get("data", {})
                        delta_content = data.get("delta_content", "")
                        phase = data.get("phase")

                        if phase == "thinking":
                            if not settings.SHOW_THINK_TAGS:
                                continue
                            if not think_open:
                                think_open = True
                                yield self._sse_chunk(completion_id, req.model, "<think>")
                            cleaned = self._clean_thinking_content(delta_content)
                            if cleaned:
                                yield self._sse_chunk(completion_id, req.model, cleaned)

                        elif phase == "answer":
                            # 如果剛從 thinking 轉入 answer，先把 </think> 前置到第一段答案
                            prefix = ""
                            if think_open:
                                prefix = "</think>"
                                think_open = False
                            cleaned = self._clean_answer_content(delta_content)
                            if cleaned or prefix:
                                yield self._sse_chunk(completion_id, req.model, f"{prefix}{cleaned}")

            # 若意外結束
            yield self._done_sse(completion_id, req.model)

        except httpx.RequestError as e:
            logger.error(e)
            yield self._error_sse(f"chatcmpl-{uuid.uuid4().hex[:29]}", req.model, f"Connection error: {e}")
        finally:
            if client:
                await client.aclose()

    # -------------------- 非串流處理 --------------------
    async def non_stream_proxy_response(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        client = None
        try:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, read=300.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                http2=True,
            )
            body, headers, cookie = await self._prepare_upstream(req)

            think_buf, answer_buf = [], []
            async with client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(cookie)
                    raise HTTPException(resp.status_code, "Upstream error")

                async for raw in resp.aiter_text():
                    for line in filter(None, (x.strip() for x in raw.split("\n"))):
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        try:
                            parsed = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        data = parsed.get("data", {})
                        phase = data.get("phase")
                        delta_content = data.get("delta_content", "")
                        if phase == "thinking":
                            think_buf.append(delta_content)
                        elif phase == "answer":
                            answer_buf.append(delta_content)

            # 組裝最終文字
            answer_text = self._clean_answer_content("".join(answer_buf))
            if settings.SHOW_THINK_TAGS and think_buf:
                think_text = self._clean_thinking_content("".join(think_buf))
                final_text = f"<think>{think_text}</think>{answer_text}"
            else:
                final_text = answer_text

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
                created=int(time.time()),
                model=req.model,
                choices=[{
                    "index": 0,
                    "message": {"role": "assistant", "content": final_text},
                    "finish_reason": "stop"
                }],
            )

        except Exception as e:
            logger.error(e)
            raise HTTPException(500, "Internal server error")
        finally:
            if client:
                await client.aclose()

    # -------------------- FastAPI 入口 --------------------
    async def handle_chat_completion(self, request: ChatCompletionRequest):
        stream = bool(request.stream) if request.stream is not None else settings.DEFAULT_STREAM
        if stream:
            return StreamingResponse(
                self.stream_proxy_response(request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return await self.non_stream_proxy_response(request)

    # -------------------- 輔助 --------------------
    @staticmethod
    def _sse_chunk(cid: str, model: str, content: str) -> str:
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    @staticmethod
    def _done_sse(cid: str, model: str) -> str:
        done = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        return f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n"

    @staticmethod
    def _error_sse(cid: str, model: str, msg: str) -> str:
        err = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": msg}, "finish_reason": "stop"}],
        }
        return f"data: {json.dumps(err, ensure_ascii=False)}\n\ndata: [DONE]\n\n"