"""
Proxy handler for Z.AI API requests
"""

import json
import logging
import re
import time
import uuid
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
        self.client = httpx.AsyncClient(timeout=60.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    def transform_content(self, content: str) -> str:
        """Transform content by replacing HTML tags and optionally removing think tags"""
        if not content:
            return content

        logger.debug(f"SHOW_THINK_TAGS setting: {settings.SHOW_THINK_TAGS}")
        logger.debug(f"Original content length: {len(content)}")
        logger.debug(f"Original content preview: {content[:200]}...")

        # Optionally remove thinking content based on configuration
        if not settings.SHOW_THINK_TAGS:
            logger.debug("Removing thinking content from response")
            original_length = len(content)

            # Remove <details> blocks (thinking content) - handle both closed and unclosed tags
            # First try to remove complete <details>...</details> blocks
            content = re.sub(
                r"<details[^>]*>.*?</details>", "", content, flags=re.DOTALL
            )

            # Then remove any remaining <details> opening tags and everything after them until we hit answer content
            # Look for pattern: <details...><summary>...</summary>...content... and remove the thinking part
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

            # Replace <details> with <think>
            content = re.sub(r"<details[^>]*>", "<think>", content)
            content = content.replace("</details>", "</think>")

            # Remove <summary> tags and their content
            content = re.sub(r"<summary>.*?</summary>", "", content, flags=re.DOTALL)

            # If there's no closing </think>, add it at the end of thinking content
            if "<think>" in content and "</think>" not in content:
                # Find where thinking ends and answer begins
                think_start = content.find("<think>")
                if think_start != -1:
                    # Look for the start of the actual answer (usually starts with a capital letter or number)
                    answer_match = re.search(r"\n\s*[A-Z0-9]", content[think_start:])
                    if answer_match:
                        insert_pos = think_start + answer_match.start()
                        content = (
                            content[:insert_pos] + "</think>\n" + content[insert_pos:]
                        )
                    else:
                        content += "</think>"

        logger.debug(f"Final transformed content length: {len(content)}")
        logger.debug(f"Final transformed content preview: {content[:200]}...")
        return content.strip()

    async def proxy_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Proxy request to Z.AI API"""
        cookie = await cookie_manager.get_next_cookie()
        if not cookie:
            raise HTTPException(status_code=503, detail="No available cookies")

        # Transform model name
        target_model = (
            settings.UPSTREAM_MODEL
            if request.model == settings.MODEL_NAME
            else request.model
        )

        # Determine if this should be a streaming response
        is_streaming = (
            request.stream if request.stream is not None else settings.DEFAULT_STREAM
        )

        # Prepare request data
        request_data = request.model_dump(exclude_none=True)

        # Build request data based on actual Z.AI format from zai-messages.md
        request_data = {
            "stream": True,  # Always request streaming from Z.AI for processing
            "model": target_model,
            "messages": request_data["messages"],
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
            response = await self.client.post(
                settings.UPSTREAM_URL, json=request_data, headers=headers
            )

            if response.status_code == 401:
                await cookie_manager.mark_cookie_failed(cookie)
                raise HTTPException(status_code=401, detail="Invalid authentication")

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Upstream error: {response.text}",
                )

            await cookie_manager.mark_cookie_success(cookie)
            return {"response": response, "cookie": cookie}

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            logger.error(f"Request error type: {type(e).__name__}")
            logger.error(f"Request URL: {settings.UPSTREAM_URL}")
            logger.error(f"Request timeout: {self.client.timeout}")
            await cookie_manager.mark_cookie_failed(cookie)
            raise HTTPException(
                status_code=503, detail=f"Upstream service unavailable: {str(e)}"
            )

    async def process_streaming_response(
        self, response: httpx.Response
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process streaming response from Z.AI"""
        buffer = ""

        async for chunk in response.aiter_text():
            buffer += chunk
            lines = buffer.split("\n")
            buffer = lines[-1]  # Keep incomplete line in buffer

            for line in lines[:-1]:
                line = line.strip()
                if not line.startswith("data: "):
                    continue

                payload = line[6:].strip()
                if payload == "[DONE]":
                    return

                try:
                    parsed = json.loads(payload)
                    logger.debug(f"Parsed chunk: {parsed}")

                    # Check for errors first
                    if parsed.get("error") or (parsed.get("data", {}).get("error")):
                        error_detail = (
                            parsed.get("error", {}).get("detail")
                            or parsed.get("data", {}).get("error", {}).get("detail")
                            or "Unknown error from upstream"
                        )
                        logger.error(f"Upstream error: {error_detail}")
                        raise HTTPException(
                            status_code=400, detail=f"Upstream error: {error_detail}"
                        )

                    # Transform the response
                    if parsed.get("data"):
                        # Remove unwanted fields
                        parsed["data"].pop("edit_index", None)
                        parsed["data"].pop("edit_content", None)

                    yield parsed

                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse JSON: {line}, error: {e}")
                    continue  # Skip non-JSON lines

    async def stream_response(
        self, response: httpx.Response, model: str
    ) -> AsyncGenerator[str, None]:
        """Generate OpenAI-compatible streaming response"""
        try:
            chunk_id = f"chatcmpl-{uuid.uuid4()}"
            
            async for parsed in self.process_streaming_response(response):
                # 取得增量內容
                delta_content = parsed.get("data", {}).get("delta_content", "")
                phase = parsed.get("data", {}).get("phase", "")
                
                logger.debug(f"Processing chunk - phase: {phase}, content: {delta_content[:50]}...")
                
                # 如果沒有內容就跳過
                if not delta_content:
                    continue
                
                # 根據 SHOW_THINK_TAGS 設定決定是否輸出
                should_output = True
                
                if not settings.SHOW_THINK_TAGS:
                    # 如果不顯示思考標籤，只輸出答案階段的內容
                    if phase != "answer":
                        logger.debug(f"Skipping non-answer phase: {phase}")
                        should_output = False
                else:
                    # 如果顯示思考標籤，對思考內容進行轉換
                    if phase == "thinking":
                        # 將 <details> 轉換為 <think>
                        delta_content = delta_content.replace("<details", "<think")
                        delta_content = delta_content.replace("</details>", "</think>")
                        # 移除 <summary> 標籤
                        delta_content = re.sub(r"<summary>.*?</summary>", "", delta_content, flags=re.DOTALL)
                
                if should_output and delta_content:
                    # 建立 OpenAI 格式的 chunk
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta_content},
                                "finish_reason": None,
                            }
                        ],
                    }
                    
                    chunk_json = json.dumps(chunk, ensure_ascii=False)
                    logger.debug(f"Yielding chunk: {chunk_json}")
                    yield f"data: {chunk_json}\n\n"
            
            # 發送完成標記
            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            # 發送錯誤訊息
            error_chunk = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"Error: {str(e)}"},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    async def handle_chat_completion(self, request: ChatCompletionRequest):
        """Handle chat completion request"""
        logger.info(f"Handling chat completion request - stream: {request.stream}, SHOW_THINK_TAGS: {settings.SHOW_THINK_TAGS}")
        
        proxy_result = await self.proxy_request(request)
        response = proxy_result["response"]

        # Determine final streaming mode
        is_streaming = (
            request.stream if request.stream is not None else settings.DEFAULT_STREAM
        )

        logger.info(f"Final streaming mode: {is_streaming}")

        if is_streaming:
            # For streaming responses
            return StreamingResponse(
                self.stream_response(response, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # 對 nginx 有用
                },
            )
        else:
            # For non-streaming responses, SHOW_THINK_TAGS setting applies
            return await self.non_stream_response(response, request.model)

    async def non_stream_response(
        self, response: httpx.Response, model: str
    ) -> ChatCompletionResponse:
        """Generate non-streaming response"""
        chunks = []
        async for parsed in self.process_streaming_response(response):
            chunks.append(parsed)
            logger.debug(f"Received chunk: {parsed}")

        if not chunks:
            raise HTTPException(status_code=500, detail="No response from upstream")

        logger.info(f"Total chunks received: {len(chunks)}")
        logger.debug(f"First chunk structure: {chunks[0] if chunks else 'None'}")

        # Aggregate content based on SHOW_THINK_TAGS setting
        full_content = ""
        
        for chunk in chunks:
            delta_content = chunk.get("data", {}).get("delta_content", "")
            phase = chunk.get("data", {}).get("phase", "")
            
            if settings.SHOW_THINK_TAGS:
                # Include all content
                full_content += delta_content
            else:
                # Only include answer phase content
                if phase == "answer":
                    full_content += delta_content

        logger.info(f"Aggregated content length: {len(full_content)}")
        logger.debug(f"Full aggregated content preview: {full_content[:200]}...")

        # Apply content transformation (including think tag filtering)
        transformed_content = self.transform_content(full_content)

        logger.info(f"Transformed content length: {len(transformed_content)}")
        logger.debug(f"Transformed content preview: {transformed_content[:200]}...")

        # Create OpenAI-compatible response
        return ChatCompletionResponse(
            id=chunks[0].get("data", {}).get("id", "chatcmpl-unknown"),
            created=int(time.time()),
            model=model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": transformed_content},
                    "finish_reason": "stop",
                }
            ],
        )