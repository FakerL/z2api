"""
Proxy handler for Z.AI API requests
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
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=300.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            http2=True,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    # ------------------------------------------------------------------
    # 內容處理相關工具
    # ------------------------------------------------------------------
    def _balance_think_tag(self, text: str) -> str:
        """確保 <think> 與 </think> 標籤數量相符"""
        open_cnt = len(re.findall(r"<think>", text))
        close_cnt = len(re.findall(r"</think>", text))
        if open_cnt > close_cnt:
            text += "</think>" * (open_cnt - close_cnt)
        elif close_cnt > open_cnt:
            extra_closes = close_cnt - open_cnt
            for _ in range(extra_closes):
                text = re.sub(r"</think>(?!</think>)(?![^<]*</think>)$", "", text, count=1)
        return text

    def _clean_thinking_content(self, content: str) -> str:
        """移除 HTML 標籤但保留思考文字"""
        if not content:
            return content
        content = re.sub(r'<details[^>]*>', '', content)
        content = re.sub(r'</details>', '', content)
        content = re.sub(r'<summary[^>]*>.*?</summary>', '', content, flags=re.DOTALL)
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'^>\s*', '', content)
        content = re.sub(r'\n>\s*', '\n', content)
        return content.strip()

    def _clean_answer_content(self, content: str) -> str:
        """僅移除 <details> 區塊，保留其餘 Markdown"""
        if not content:
            return content
        return re.sub(r"<details[^>]*>.*?</details>", "", content, flags=re.DOTALL)

    def transform_content(self, content: str) -> str:
        """將上游 HTML 轉成 <think> 標籤"""
        if not content:
            return content
        if settings.SHOW_THINK_TAGS:
            content = re.sub(r"<details[^>]*>", "<think>", content)
            content = re.sub(r"</details>", "</think>", content)
            content = re.sub(r"<summary>.*?</summary>", "", content, flags=re.DOTALL)
            content = self._balance_think_tag(content)
        else:
            content = re.sub(r"<details[^>]*>.*?</details>", "", content, flags=re.DOTALL)
        return content.strip()

    def _serialize_messages(self, messages) -> list:
        """將 ChatMessage 物件序列化成 dict 方便 JSON 傳輸"""
        serialized_messages = []
        for message in messages:
            if hasattr(message, "dict"):
                serialized_messages.append(message.dict())
            elif hasattr(message, "model_dump"):
                serialized_messages.append(message.model_dump())
            elif isinstance(message, dict):
                serialized_messages.append(message)
            else:
                serialized_messages.append(
                    {
                        "role": getattr(message, "role", "user"),
                        "content": getattr(message, "content", str(message)),
                    }
                )
        return serialized_messages

    # ------------------------------------------------------------------
    # 與上游 API 溝通
    # ------------------------------------------------------------------
    async def _prepare_upstream(self, request: ChatCompletionRequest) -> Tuple[Dict[str, Any], Dict[str, str], str]:
        """產生送往上游的 body、headers 與 cookie"""
        cookie = await cookie_manager.get_next_cookie()
        if not cookie:
            raise HTTPException(status_code=503, detail="No available cookies")

        target_model = settings.UPSTREAM_MODEL if request.model == settings.MODEL_NAME else request.model

        req_body = {
            "stream": True,
            "model": target_model,
            "messages": self._serialize_messages(request.messages),
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

        return req_body, headers, cookie

    # ------------------------------------------------------------------
    # 串流模式
    # ------------------------------------------------------------------
    async def stream_proxy_response(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """處理串流回傳"""
        client = None
        try:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, read=300.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                http2=True,
            )
            req_body, headers, cookie = await self._prepare_upstream(request)
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"

            think_open = False          # 是否已開啟 <think>
            current_phase = None        # 追蹤 phase 切換

            async with client.stream("POST", settings.UPSTREAM_URL, json=req_body, headers=headers) as response:
                if response.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(cookie)
                    error_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": f"Error: Upstream API returned {response.status_code}"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for raw_chunk in response.aiter_text():
                    if not raw_chunk or raw_chunk.isspace():
                        continue

                    for line in raw_chunk.split("\n"):
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data: "):
                            continue

                        payload = line[6:]
                        if payload == "[DONE]":
                            # 結尾時若 <think> 未關閉，自動補上
                            if think_open and settings.SHOW_THINK_TAGS:
                                close_chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": request.model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": "</think>\n"},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(close_chunk)}\n\n"

                            final_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [
                                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                                ],
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        try:
                            parsed = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        data = parsed.get("data", {})
                        delta_content = data.get("delta_content", "")
                        phase = data.get("phase")

                        # --------------------------------------------------
                        # phase 變化處理
                        # --------------------------------------------------
                        if phase != current_phase:
                            current_phase = phase
                            # 由 thinking → answer 時，先補 </think> 並帶換行
                            if (
                                phase == "answer"
                                and think_open
                                and settings.SHOW_THINK_TAGS
                            ):
                                auto_close = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": request.model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": "</think>\n"},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(auto_close)}\n\n"
                                think_open = False

                        # --------------------------------------------------
                        # 真正內容輸出
                        # --------------------------------------------------
                        if phase == "thinking" and not settings.SHOW_THINK_TAGS:
                            # 不顯示思考內容就跳過
                            continue

                        if delta_content:
                            if phase == "thinking" and settings.SHOW_THINK_TAGS:
                                # 若第一次進入思考階段，先輸出 <think>
                                if not think_open:
                                    think_chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": request.model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": "<think>"},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                    yield f"data: {json.dumps(think_chunk)}\n\n"
                                    think_open = True

                                transformed_content = self._clean_thinking_content(delta_content)
                            else:
                                # answer 階段
                                transformed_content = self._clean_answer_content(delta_content)

                            if transformed_content:
                                chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": request.model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": transformed_content},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            error_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"Connection error: {str(e)}"},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Unexpected error in streaming: {e}")
            error_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"Internal error: {str(e)}"},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        finally:
            if client:
                await client.aclose()

    # ------------------------------------------------------------------
    # 非串流模式
    # ------------------------------------------------------------------
    async def non_stream_proxy_response(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """處理非串流回傳"""
        client = None
        try:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, read=300.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                http2=True,
            )
            req_body, headers, cookie = await self._prepare_upstream(request)

            thinking_buf, answer_buf = [], []
            current_phase = None

            async with client.stream("POST", settings.UPSTREAM_URL, json=req_body, headers=headers) as response:
                if response.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(cookie)
                    raise HTTPException(status_code=response.status_code, detail="Upstream API error")

                async for raw_chunk in response.aiter_text():
                    if not raw_chunk or raw_chunk.isspace():
                        continue

                    for line in raw_chunk.split("\n"):
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

                        if delta_content:
                            if phase == "thinking":
                                thinking_buf.append(delta_content)
                            elif phase == "answer":
                                answer_buf.append(delta_content)

            # 組裝最終文字
            final_text = ""
            if settings.SHOW_THINK_TAGS and thinking_buf:
                thinking_text = self._clean_thinking_content("".join(thinking_buf))
                answer_text = self._clean_answer_content("".join(answer_buf)) if answer_buf else ""
                if thinking_text:
                    # 在 </think> 與正文間補換行，避免首字被吃掉
                    final_text = f"<think>{thinking_text}</think>\n{answer_text}"
                else:
                    final_text = answer_text
            else:
                final_text = self._clean_answer_content("".join(answer_buf))

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": final_text},
                        "finish_reason": "stop",
                    }
                ],
            )

        except httpx.RequestError as e:
            logger.error(f"Request error in non-stream: {e}")
            if "cookie" in locals():
                await cookie_manager.mark_cookie_invalid(cookie)
            raise HTTPException(status_code=502, detail="Upstream connection error")

        except Exception as e:
            logger.error(f"Unexpected error in non-stream: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

        finally:
            if client:
                await client.aclose()

    # ------------------------------------------------------------------
    # FastAPI 入口
    # ------------------------------------------------------------------
    async def handle_chat_completion(self, request: ChatCompletionRequest):
        """聊天補全主入口"""
        stream = bool(request.stream) if request.stream is not None else settings.DEFAULT_STREAM
        if stream:
            return StreamingResponse(
                self.stream_proxy_response(request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            return await self.non_stream_proxy_response(request)