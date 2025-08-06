"""
Proxy handler for Z.AI API requests
"""

import json
import logging
import re
import time
from typing import AsyncGenerator, Dict, Any, Optional
import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from cookie_manager import cookie_manager
from models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)

logger = logging.getLogger(__name__)


class ProxyHandler:
    def __init__(self):
        # Configure httpx client for streaming support
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=300.0),  # Longer read timeout for streaming
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            http2=True  # Enable HTTP/2 for better streaming performance
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def handle_chat_completion(self, request: ChatCompletionRequest):
        """Handle chat completion request"""
        # Determine final streaming mode
        is_streaming = (
            request.stream if request.stream is not None else settings.DEFAULT_STREAM
        )

        if is_streaming:
            # For streaming responses, use direct streaming proxy
            return StreamingResponse(
                self.stream_proxy_response(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # For non-streaming, collect and process the full response
            return await self.aggregate_non_stream_response(request)

    async def _build_request_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Helper to build the Z.AI request payload"""
        import uuid
        target_model = "0727-360B-API"
        
        return {
            "stream": True,
            "model": target_model,
            "messages": request.messages,
            "background_tasks": {
                "title_generation": True,
                "tags_generation": True
            },
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
            "model_item": {
                "id": target_model,
                "name": "GLM-4.5",
                "owned_by": "openai"
            },
            "params": {},
            "tool_servers": [],
            "variables": {
                "{{USER_NAME}}": "User",
                "{{USER_LOCATION}}": "Unknown",
                "{{CURRENT_DATETIME}}": "2025-08-04 16:46:56"
            }
        }
        
    def _build_request_headers(self, cookie: str) -> Dict[str, str]:
        """Helper to build request headers"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cookie}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
            "Accept": "application/json, text/event-stream",
            "Accept-Language": "zh-CN",
            "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "x-fe-version": "prod-fe-1.0.53",
            "Origin": "https://chat.z.ai",
            "Referer": "https://chat.z.ai/",
        }

    async def stream_proxy_response(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Handle true streaming response, transforming Z.AI stream to OpenAI format."""
        import uuid
        
        cookie = await cookie_manager.get_next_cookie()
        if not cookie:
            raise HTTPException(status_code=503, detail="No valid authentication available")

        request_data = await self._build_request_payload(request)
        headers = self._build_request_headers(cookie)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        
        is_in_think_block = False

        try:
            async with self.client.stream("POST", settings.UPSTREAM_URL, json=request_data, headers=headers) as response:
                if response.status_code == 401:
                    await cookie_manager.mark_cookie_failed(cookie)
                    raise HTTPException(status_code=401, detail="Invalid authentication")
                if response.status_code != 200:
                    await cookie_manager.mark_cookie_failed(cookie)
                    error_text = await response.aread()
                    raise HTTPException(status_code=response.status_code, detail=f"Upstream error: {error_text.decode()}")

                await cookie_manager.mark_cookie_success(cookie)

                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line.startswith("data:"):
                            continue

                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            break
                        
                        try:
                            parsed = json.loads(payload)
                            data = parsed.get("data", {})
                            delta_content = data.get("delta_content", "")
                            phase = data.get("phase", "")

                            content_to_send = ""

                            if settings.SHOW_THINK_TAGS:
                                # State-based think tag management
                                if phase == "thinking" and not is_in_think_block:
                                    content_to_send += "<think>"
                                    is_in_think_block = True
                                elif phase == "answer" and is_in_think_block:
                                    content_to_send += "</think>\n"
                                    is_in_think_block = False
                                
                                # Clean up original tags regardless of phase
                                cleaned_delta = re.sub(r"<details[^>]*>.*?</details>", "", delta_content, flags=re.DOTALL)
                                cleaned_delta = re.sub(r"<summary>.*?</summary>", "", cleaned_delta, flags=re.DOTALL)
                                content_to_send += cleaned_delta

                            else: # SHOW_THINK_TAGS is false
                                if phase == "answer":
                                    # Still need to clean the first answer chunk which might contain </details>
                                    cleaned_delta = re.sub(r"</details>", "", delta_content)
                                    content_to_send += cleaned_delta
                            
                            if content_to_send:
                                openai_chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": request.model,
                                    "choices": [{"index": 0, "delta": {"content": content_to_send}, "finish_reason": None}]
                                }
                                yield f"data: {json.dumps(openai_chunk)}\n\n"

                        except json.JSONDecodeError:
                            continue

        except httpx.RequestError as e:
            await cookie_manager.mark_cookie_failed(cookie)
            raise HTTPException(status_code=503, detail=f"Upstream service unavailable: {str(e)}")
        finally:
            if is_in_think_block:
                # Ensure the think tag is closed if stream ends unexpectedly
                closing_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": "\n</think>"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(closing_chunk)}\n\n"

            # Send final completion chunk
            final_chunk = {
                "id": completion_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"


    async def aggregate_non_stream_response(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle non-streaming response by aggregating the stream."""
        import uuid

        cookie = await cookie_manager.get_next_cookie()
        if not cookie:
            raise HTTPException(status_code=503, detail="No valid authentication available")

        request_data = await self._build_request_payload(request)
        headers = self._build_request_headers(cookie)
        
        content_parts = []
        is_in_think_block = False

        try:
            async with self.client.stream("POST", settings.UPSTREAM_URL, json=request_data, headers=headers) as response:
                if response.status_code == 401:
                    await cookie_manager.mark_cookie_failed(cookie)
                    raise HTTPException(status_code=401, detail="Invalid authentication")
                if response.status_code != 200:
                    await cookie_manager.mark_cookie_failed(cookie)
                    error_text = await response.aread()
                    raise HTTPException(status_code=response.status_code, detail=f"Upstream error: {error_text.decode()}")

                await cookie_manager.mark_cookie_success(cookie)

                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line.startswith("data:"):
                            continue

                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            break
                        
                        try:
                            parsed = json.loads(payload)
                            data = parsed.get("data", {})
                            delta_content = data.get("delta_content", "")
                            phase = data.get("phase", "")

                            if settings.SHOW_THINK_TAGS:
                                if phase == "thinking" and not is_in_think_block:
                                    content_parts.append("<think>")
                                    is_in_think_block = True
                                elif phase == "answer" and is_in_think_block:
                                    content_parts.append("</think>\n")
                                    is_in_think_block = False
                                
                                cleaned_delta = re.sub(r"<details[^>]*>.*?</details>", "", delta_content, flags=re.DOTALL)
                                cleaned_delta = re.sub(r"<summary>.*?</summary>", "", cleaned_delta, flags=re.DOTALL)
                                content_parts.append(cleaned_delta)
                            
                            else: # SHOW_THINK_TAGS is false
                                if phase == "answer":
                                    cleaned_delta = re.sub(r"</details>", "", delta_content)
                                    content_parts.append(cleaned_delta)

                        except json.JSONDecodeError:
                            continue
            
            if is_in_think_block:
                content_parts.append("</think>")

        except httpx.RequestError as e:
            await cookie_manager.mark_cookie_failed(cookie)
            raise HTTPException(status_code=503, detail=f"Upstream service unavailable: {str(e)}")

        full_content = "".join(content_parts).strip()

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_content},
                    "finish_reason": "stop",
                }
            ],
        )