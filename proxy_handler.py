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
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=300.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            http2=True
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    def transform_content(self, content: str) -> str:
        if not content:
            return content

        logger.debug(f"SHOW_THINK_TAGS setting: {settings.SHOW_THINK_TAGS}")

        if not settings.SHOW_THINK_TAGS:
            logger.debug("Removing thinking content from response")
            original_length = len(content)
            content = re.sub(
                r"<details[^>]*>.*?</details>", "", content, flags=re.DOTALL
            )
            content = re.sub(
                r"<details[^>]*>.*?(?=\s*[A-Z]|\s*\d|\s*$)",
                "",
                content,
                flags=re.DOTALL,
            )
            content = content.strip()
            logger.debug(
                f"Content length after removing thinking content: {original_length} -> {len(content)}"
            )
        else:
            logger.debug("Keeping thinking content, converting to <think> tags")
            content = re.sub(r"<details[^>]*>", "<think>", content)
            content = content.replace("</details>", "</think>")
            content = re.sub(r"<summary>.*?</summary>", "", content, flags=re.DOTALL)
            if "<think>" in content and "</think>" not in content:
                think_start = content.find("<think>")
                if think_start != -1:
                    answer_match = re.search(r"\n\s*[A-Z0-9]", content[think_start:])
                    if answer_match:
                        insert_pos = think_start + answer_match.start()
                        content = (
                            content[:insert_pos] + "</think>\n" + content[insert_pos:]
                        )
                    else:
                        content += "</think>"

        return content.strip()

    async def proxy_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        cookie = await cookie_manager.get_next_cookie()
        if not cookie:
            raise HTTPException(status_code=503, detail="No available cookies")

        target_model = (
            settings.UPSTREAM_MODEL
            if request.model == settings.MODEL_NAME
            else request.model
        )
        
        is_streaming = (
            request.stream if request.stream is not None else settings.DEFAULT_STREAM
        )

        if is_streaming and not settings.SHOW_THINK_TAGS:
            logger.warning("SHOW_THINK_TAGS=false is ignored for streaming responses")

        import uuid
        request_data = {
            "stream": True,
            "model": target_model,
            "messages": request.model_dump(exclude_none=True)["messages"],
            "background_tasks": {"title_generation": True, "tags_generation": True},
            "chat_id": str(uuid.uuid4()),
            "features": {
                "image_generation": False,
                "code_interpreter": False,
                "web_search": False,
                "auto_web_search": False,
            },
            "id": str(uuid.uuid4()),
            "mcp_servers": ["deep-web-search"],
            "model_item": {"id": target_model, "name": "GLM-4.5", "owned_by": "openai"},
            "params": {},
            "tool_servers": [],
            "variables": {
                "{{USER_NAME}}": "User",
                "{{USER_LOCATION}}": "Unknown",
                "{{CURRENT_DATETIME}}": "2025-08-04 16:46:56",
            },
        }

        logger.debug(f"Sending request data: {request_data}")

        headers = {
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
            "Referer": "https://chat.z.ai/c/069723d5-060b-404f-992c-4705f1554c4c",
        }

        try:
            async with self.client.stream(
                "POST",
                settings.UPSTREAM_URL,
                json=request_data,
                headers=headers,
                timeout=httpx.Timeout(60.0, read=300.0)
            ) as response:
                if response.status_code == 401:
                    await cookie_manager.mark_cookie_failed(cookie)
                    raise HTTPException(status_code=401, detail="Invalid authentication")
                if response.status_code != 200:
                    try:
                        error_text = await response.aread()
                        error_detail = error_text.decode('utf-8')
                    except:
                        error_detail = f"HTTP {response.status_code}"
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Upstream error: {error_detail}",
                    )
                await cookie_manager.mark_cookie_success(cookie)
                return {"response": response, "cookie": cookie}
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            await cookie_manager.mark_cookie_failed(cookie)
            raise HTTPException(
                status_code=503, detail=f"Upstream service unavailable: {str(e)}"
            )

    async def handle_chat_completion(self, request: ChatCompletionRequest):
        is_streaming = (
            request.stream if request.stream is not None else settings.DEFAULT_STREAM
        )

        if is_streaming:
            return StreamingResponse(
                self.stream_proxy_response(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            full_content = ""
            async for chunk_data in self.stream_proxy_response(request):
                if chunk_data.startswith("data: ") and not chunk_data.startswith("data: [DONE]"):
                    try:
                        chunk_json_str = chunk_data[6:]
                        if not chunk_json_str.strip():
                            continue
                        chunk_json = json.loads(chunk_json_str)
                        delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                        if delta and delta.get("content"):
                            full_content += delta["content"]
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode JSON from stream chunk: {chunk_data}")
                        continue
            
            import uuid
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": full_content.strip()},
                        "finish_reason": "stop",
                    }
                ],
            )

    async def stream_proxy_response(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        import uuid
        
        proxy_result = await self.proxy_request(request)
        response = proxy_result["response"]
        
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        in_thinking_phase = False
        buffer = ""

        try:
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if not line.startswith("data:"):
                        continue
                    
                    payload_str = line[6:].strip()
                    if payload_str == "[DONE]":
                        if in_thinking_phase and settings.SHOW_THINK_TAGS:
                            closing_think_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{"index": 0, "delta": {"content": "</think>"}, "finish_reason": None}]
                            }
                            yield f"data: {json.dumps(closing_think_chunk)}\n\n"
                        break

                    try:
                        parsed = json.loads(payload_str)
                        data = parsed.get("data", {})
                        delta_content = data.get("delta_content", "")
                        phase = data.get("phase", "").strip()
                        output_content = ""

                        # --- State Transition and Content Processing ---
                        
                        # 1. Entering 'thinking' phase
                        if phase == "thinking" and not in_thinking_phase:
                            in_thinking_phase = True
                            if settings.SHOW_THINK_TAGS:
                                output_content += "<think>"
                        
                        # 2. Exiting 'thinking' phase
                        elif phase != "thinking" and in_thinking_phase:
                            in_thinking_phase = False
                            if settings.SHOW_THINK_TAGS:
                                output_content += "</think>"
                        
                        # 3. Process content based on phase
                        if phase == "thinking":
                            if settings.SHOW_THINK_TAGS:
                                # Clean up Z.AI's HTML boilerplate from thinking chunks
                                cleaned_delta = re.sub(r"<details[^>]*>|<summary>.*?</summary>", "", delta_content, flags=re.DOTALL)
                                output_content += cleaned_delta
                        elif phase == "answer":
                            # Clean up potential leftover tag from Z.AI's stream
                            cleaned_delta = delta_content.replace("</details>", "")
                            output_content += cleaned_delta
                        
                        if output_content:
                            openai_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": output_content},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n"

                    except json.JSONDecodeError:
                        logger.debug(f"Skipping non-JSON data line: {line}")
                        continue
                else: # while loop exit
                    continue
                break # for loop exit after [DONE]

            # Send final completion chunk
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except httpx.RequestError as e:
            logger.error(f"Streaming request error: {e}")
            raise HTTPException(status_code=503, detail=f"Upstream service unavailable: {str(e)}")
        finally:
            await response.aclose()