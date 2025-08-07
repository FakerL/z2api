"""
Proxy handler for Z.AI API requests (patched version)
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
from models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

logger = logging.getLogger(__name__)


class ProxyHandler:
    def __init__(self):
        # Consider moving shared client to FastAPI startup
        # But keeping old logic for convenience
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=300.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            http2=True,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    # Content processing utilities

    def _balance_think_tag(self, text: str) -> str:
        """Ensure <think> and </think> tags are balanced"""
        open_cnt = len(re.findall(r"<think>", text))
        close_cnt = len(re.findall(r"</think>", text))
        if open_cnt > close_cnt:
            text += "</think>" * (open_cnt - close_cnt)
        elif close_cnt > open_cnt:
            # Remove extra </think> tags
            extra = close_cnt - open_cnt
            for _ in range(extra):
                text = re.sub(r"</think>(?!.*</think>)", "", text, count=1)
        return text

    def _clean_thinking_content(self, content: str) -> str:
        """Remove HTML tags, keep only thinking text"""
        if not content:
            return content
        content = re.sub(r"<details[^>]*>", "", content)
        content = re.sub(r"</details>", "", content)
        content = re.sub(r"<summary[^>]*>.*?</summary>", "", content, flags=re.DOTALL)
        content = re.sub(r"<[^>]+>", "", content)
        content = re.sub(r"^\s*>\s*", "", content)
        content = re.sub(r"\n\s*>\s*", "\n", content)
        return content

    def _clean_answer_content(self, content: str) -> str:
        """Remove <details>...</details>, keep the rest"""
        if not content:
            return content
        return re.sub(r"<details[^>]*>.*?</details>", "", content, flags=re.DOTALL)

    def transform_content(self, content: str) -> str:
        """Transform <details> to <think> tags if showing thinking"""
        if settings.SHOW_THINK_TAGS:
            content = re.sub(r"<details[^>]*>", "<think>", content)
            content = re.sub(r"</details>", "</think>", content)
            content = re.sub(r"<summary[^>]*>.*?</summary>", "", content, flags=re.DOTALL)
            content = self._balance_think_tag(content)
        else:
            content = re.sub(r"<details[^>]*>.*?</details>", "", content, flags=re.DOTALL)
        return content

    def _serialize_messages(self, messages) -> list:
        """Convert pydantic/dataclass/dict to pure dict"""
        result = []
        for m in messages:
            if hasattr(m, "dict"):
                result.append(m.dict())
            elif hasattr(m, "model_dump"):
                result.append(m.model_dump())
            elif isinstance(m, dict):
                result.append(m)
            else:
                result.append({
                    "role": getattr(m, "role", "user"),
                    "content": getattr(m, "content", str(m))
                })
        return result

    # Upstream communication

    async def _prepare_upstream(
        self, request: ChatCompletionRequest
    ) -> Tuple[Dict[str, Any], Dict[str, str], str]:
        cookie = await cookie_manager.get_next_cookie()
        if not cookie:
            raise HTTPException(status_code=503, detail="No available cookies")

        target_model = settings.UPSTREAM_MODEL if request.model == settings.MODEL_NAME else request.model
        serialized_messages = self._serialize_messages(request.messages)

        body = {
            "stream": True,
            "model": target_model,
            "messages": serialized_messages,
            "background_tasks": {"title_generation": True, "tags_generation": True},
            "chat_id": str(uuid.uuid4()),
            "features": {
                "image_generation": False,
                "code_interpreter": False,
                "web_search": False,
                "auto_web_search": False,
                "enable_thinking": True,
            },
            "id": str(uuid.uuid4()),
            "mcp_servers": ["deep-web-search"],
            "model_item": {"id": target_model, "name": "GLM-4.5", "owned_by": "openai"},
            "params": {},
            "tool_servers": [],
            "variables": {
                "{{USER_NAME}}": "User",
                "{{USER_LOCATION}}": "Unknown",
                "{{CURRENT_DATETIME}}": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cookie}",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/event-stream",
            "Accept-Language": "zh-CN",
            "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "x-fe-version": "prod-fe-1.0.53",
            "Origin": "https://chat.z.ai",
            "Referer": "https://chat.z.ai/",
        }
        return body, headers, cookie

    # Utilities

    def _mk_chunk(self, cid: str, model: str, content: str) -> str:
        """Create a chunk and return with data: prefix"""
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None
            }]
        }
        return f"data: {json.dumps(chunk)}\n\n"

    async def _error_chunks_async(self, model: str, msg: str):
        """Generate error output (async generator)"""
        cid = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        err = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": msg},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"

    # Streaming response

    async def stream_proxy_response(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        client = None
        cookie = None
        try:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, read=300.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                http2=True,
            )
            body, headers, cookie = await self._prepare_upstream(request)

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            think_open = False
            current_phase = None

            async with client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(cookie)
                    err = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"Error: Upstream returned {resp.status_code}"},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(err)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for raw in resp.aiter_text():
                    if not raw or raw.isspace():
                        continue
                    for line in raw.split("\n"):
                        line = line.strip()
                        if not line or line.startswith(":") or not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            # If still in thinking mode, close the tag
                            if think_open and settings.SHOW_THINK_TAGS:
                                yield self._mk_chunk(completion_id, request.model, "</think>")
                                yield self._mk_chunk(completion_id, request.model, "\n")
                            # Final stop chunk
                            final = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            yield f"data: {json.dumps(final)}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        try:
                            parsed = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        data = parsed.get("data", {})
                        delta_content = data.get("delta_content", "")
                        phase = data.get("phase")

                        # Phase change: thinking -> answer
                        if phase != current_phase:
                            current_phase = phase
                            if phase == "answer" and think_open and settings.SHOW_THINK_TAGS:
                                # Close </think> first
                                yield self._mk_chunk(completion_id, request.model, "</think>")
                                # Then send a separate newline
                                yield self._mk_chunk(completion_id, request.model, "\n")
                                think_open = False

                        if phase == "thinking" and not settings.SHOW_THINK_TAGS:
                            continue

                        if not delta_content:
                            continue

                        # Content cleaning
                        if phase == "thinking":
                            if settings.SHOW_THINK_TAGS:
                                if not think_open:
                                    # Open <think> tag
                                    yield self._mk_chunk(completion_id, request.model, "<think>\n")
                                    think_open = True
                                processed_content = self._clean_thinking_content(delta_content)
                            else:
                                continue
                        else:
                            processed_content = self._clean_answer_content(delta_content)

                        if processed_content:
                            yield self._mk_chunk(completion_id, request.model, processed_content)

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            if cookie:
                await cookie_manager.mark_cookie_invalid(cookie)
            async for chunk in self._error_chunks_async(request.model, "Upstream connection error"):
                yield chunk
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            async for chunk in self._error_chunks_async(request.model, "Internal server error"):
                yield chunk
        finally:
            if client:
                await client.aclose()

    async def non_stream_proxy_response(self, request: ChatCompletionRequest):
        client = None
        cookie = None
        try:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, read=300.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                http2=True,
            )
            body, headers, cookie = await self._prepare_upstream(request)

            thinking_text = ""
            answer_text = ""
            current_phase = None

            async with client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(cookie)
                    raise HTTPException(status_code=502, detail=f"Upstream returned {resp.status_code}")

                async for raw in resp.aiter_text():
                    if not raw or raw.isspace():
                        continue
                    for line in raw.split("\n"):
                        line = line.strip()
                        if not line or line.startswith(":") or not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break

                        try:
                            parsed = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        data = parsed.get("data", {})
                        delta_content = data.get("delta_content", "")
                        phase = data.get("phase")

                        if phase != current_phase:
                            current_phase = phase

                        if phase == "thinking" and settings.SHOW_THINK_TAGS:
                            thinking_text += self._clean_thinking_content(delta_content)
                        elif phase == "answer":
                            answer_text += self._clean_answer_content(delta_content)

            # Combine thinking and answer
            if thinking_text and settings.SHOW_THINK_TAGS:
                if answer_text:
                    final_text = f"<think>\n{thinking_text}</think>\n{answer_text}"
                else:
                    final_text = f"<think>\n{thinking_text}</think>"
            else:
                final_text = answer_text

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
                created=int(time.time()),
                model=request.model,
                choices=[{
                    "index": 0,
                    "message": {"role": "assistant", "content": final_text},
                    "finish_reason": "stop"
                }],
            )

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            if cookie:
                await cookie_manager.mark_cookie_invalid(cookie)
            raise HTTPException(status_code=502, detail="Upstream connection error")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            if client:
                await client.aclose()

    # FastAPI entry point

    async def handle_chat_completion(self, request: ChatCompletionRequest):
        stream = bool(request.stream) if request.stream is not None else settings.DEFAULT_STREAM
        if stream:
            return StreamingResponse(
                self.stream_proxy_response(request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            return await self.non_stream_proxy_response(request)