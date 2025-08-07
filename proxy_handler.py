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

    # Content transformation utilities
    def _balance_think_tag(self, text: str) -> str:
        """Ensure matching number of <think> and </think> tags"""
        open_cnt = len(re.findall(r"<think>", text))
        close_cnt = len(re.findall(r"</think>", text))
        if open_cnt > close_cnt:
            text += "</think>" * (open_cnt - close_cnt)
        elif close_cnt > open_cnt:
            # Remove extra closing tags from the end
            extra_closes = close_cnt - open_cnt
            for _ in range(extra_closes):
                text = re.sub(r"</think>(?!</think>)(?![^<]*</think>)$", "", text, count=1)
        return text

    def _minimal_clean_content(self, content: str) -> str:
        """Minimal content cleaning to preserve all characters"""
        if not content:
            return content
        
        # Log original content for debugging
        logger.debug(f"Original content: {repr(content[:100])}")
        
        # Only remove the most obvious HTML artifacts, nothing else
        # Remove <details> blocks only
        cleaned = re.sub(r'<details[^>]*>.*?</details>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove <summary> blocks only
        cleaned = re.sub(r'<summary[^>]*>.*?</summary>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # That's it - no other processing to avoid accidentally removing characters
        
        # Log cleaned content for debugging
        logger.debug(f"Cleaned content: {repr(cleaned[:100])}")
        
        return cleaned

    def _clean_thinking_content(self, content: str) -> str:
        """Clean thinking content with minimal processing"""
        return self._minimal_clean_content(content)

    def _clean_answer_content(self, content: str) -> str:
        """Clean answer content with minimal processing"""
        return self._minimal_clean_content(content)

    def transform_content(self, content: str) -> str:
        """Transform upstream HTML to <think> format"""
        if not content:
            return content
            
        if settings.SHOW_THINK_TAGS:
            # Replace <details> with <think>
            content = re.sub(r"<details[^>]*>", "<think>", content, flags=re.IGNORECASE)
            content = re.sub(r"</details>", "</think>", content, flags=re.IGNORECASE)
            # Remove <summary> tags
            content = re.sub(r"<summary>.*?</summary>", "", content, flags=re.DOTALL | re.IGNORECASE)
            content = self._balance_think_tag(content)
        else:
            # Remove entire <details> blocks
            content = re.sub(r"<details[^>]*>.*?</details>", "", content, flags=re.DOTALL | re.IGNORECASE)

        return content.strip()

    def _serialize_messages(self, messages) -> list:
        """Convert ChatMessage objects to dict format for JSON serialization"""
        serialized_messages = []
        for message in messages:
            if hasattr(message, 'dict'):
                # Pydantic model
                serialized_messages.append(message.dict())
            elif hasattr(message, 'model_dump'):
                # Pydantic v2 model
                serialized_messages.append(message.model_dump())
            elif isinstance(message, dict):
                # Already a dict
                serialized_messages.append(message)
            else:
                # Try to convert to dict
                serialized_messages.append({
                    "role": getattr(message, 'role', 'user'),
                    "content": getattr(message, 'content', str(message))
                })
        return serialized_messages

    # Upstream communication
    async def _prepare_upstream(self, request: ChatCompletionRequest) -> Tuple[Dict[str, Any], Dict[str, str], str]:
        """Prepare request body and headers for upstream API"""
        cookie = await cookie_manager.get_next_cookie()
        if not cookie:
            raise HTTPException(status_code=503, detail="No available cookies")

        target_model = settings.UPSTREAM_MODEL if request.model == settings.MODEL_NAME else request.model

        # Serialize messages to dict format
        serialized_messages = self._serialize_messages(request.messages)

        req_body = {
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
        return req_body, headers, cookie

    # Main streaming logic
    async def stream_proxy_response(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Handle streaming proxy response with minimal processing"""
        client = None
        try:
            # Create a new client for this request
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, read=300.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                http2=True,
            )
            
            req_body, headers, cookie = await self._prepare_upstream(request)
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            think_open = False  # Track if <think> tag is open
            current_phase = None

            async with client.stream("POST", settings.UPSTREAM_URL, json=req_body, headers=headers) as response:
                if response.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(cookie)
                    error_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"Error: Upstream API returned {response.status_code}"},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Accumulate raw text to ensure we don't split events incorrectly
                raw_buffer = ""
                
                async for raw_chunk in response.aiter_text():
                    if not raw_chunk:
                        continue

                    raw_buffer += raw_chunk
                    
                    # Process complete events only
                    while '\n\n' in raw_buffer:
                        event_block, raw_buffer = raw_buffer.split('\n\n', 1)
                        
                        # Process each line in the event block
                        for line in event_block.split('\n'):
                            line = line.strip()
                            if not line or line.startswith(':') or not line.startswith('data: '):
                                continue

                            payload = line[6:]  # Remove 'data: '
                            
                            if payload == '[DONE]':
                                # Close thinking tag if still open
                                if think_open and settings.SHOW_THINK_TAGS:
                                    close_chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": request.model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": "</think>"},
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(close_chunk)}\n\n"

                                final_chunk = {
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
                                yield f"data: {json.dumps(final_chunk)}\n\n"
                                yield "data: [DONE]\n\n"
                                return

                            # Parse JSON payload
                            try:
                                parsed = json.loads(payload)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse JSON: {e}, payload: {payload[:100]}")
                                continue

                            data = parsed.get("data", {})
                            delta_content = data.get("delta_content", "")
                            phase = data.get("phase")

                            # Handle phase changes
                            if phase != current_phase:
                                current_phase = phase
                                if phase == "answer" and think_open and settings.SHOW_THINK_TAGS:
                                    # Auto-close thinking phase
                                    auto_close = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": request.model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": "</think>"},
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(auto_close)}\n\n"
                                    think_open = False

                            # Process delta content
                            if delta_content:
                                logger.debug(f"Raw delta_content ({phase}): {repr(delta_content[:50])}")
                                
                                # Skip thinking content if not showing think tags
                                if phase == "thinking" and not settings.SHOW_THINK_TAGS:
                                    continue

                                if phase == "thinking" and settings.SHOW_THINK_TAGS:
                                    # First time entering thinking phase, add <think> tag
                                    if not think_open:
                                        think_chunk = {
                                            "id": completion_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": request.model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": "<think>"},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {json.dumps(think_chunk)}\n\n"
                                        think_open = True

                                # Apply ABSOLUTE MINIMAL processing
                                if phase == "thinking":
                                    # For thinking: only remove obvious HTML tags, keep everything else
                                    output_content = delta_content
                                    # Only remove details/summary tags if they exist
                                    if '<details' in output_content or '<summary' in output_content:
                                        output_content = self._clean_thinking_content(output_content)
                                else:
                                    # For answer: even more minimal processing
                                    output_content = delta_content
                                    # Only remove details tags if they exist
                                    if '<details' in output_content:
                                        output_content = self._clean_answer_content(output_content)

                                # Log the final output for debugging
                                if output_content:
                                    logger.debug(f"Output content ({phase}): {repr(output_content[:50])}")
                                    
                                    chunk = {
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
                                    yield f"data: {json.dumps(chunk)}\n\n"

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            error_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"Connection error: {str(e)}"},
                    "finish_reason": "stop"
                }]
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
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"Internal error: {str(e)}"},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if client:
                await client.aclose()

    async def non_stream_proxy_response(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle non-streaming proxy response with minimal processing"""
        client = None
        try:
            # Create a new client for this request
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, read=300.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                http2=True,
            )
            
            req_body, headers, cookie = await self._prepare_upstream(request)
            thinking_buf = []
            answer_buf = []
            current_phase = None
            raw_buffer = ""

            async with client.stream("POST", settings.UPSTREAM_URL, json=req_body, headers=headers) as response:
                if response.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(cookie)
                    raise HTTPException(status_code=response.status_code, detail="Upstream API error")

                async for raw_chunk in response.aiter_text():
                    if not raw_chunk:
                        continue

                    raw_buffer += raw_chunk
                    
                    # Process complete events only
                    while '\n\n' in raw_buffer:
                        event_block, raw_buffer = raw_buffer.split('\n\n', 1)
                        
                        for line in event_block.split('\n'):
                            line = line.strip()
                            if not line or line.startswith(':') or not line.startswith('data: '):
                                continue

                            payload = line[6:]
                            if payload == '[DONE]':
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
                                logger.debug(f"Non-stream raw delta_content ({phase}): {repr(delta_content[:50])}")
                                
                                if phase == "thinking":
                                    thinking_buf.append(delta_content)
                                elif phase == "answer":
                                    answer_buf.append(delta_content)

            # Combine content with minimal processing
            final_text = ""
            
            if settings.SHOW_THINK_TAGS and thinking_buf:
                # Join raw thinking content
                thinking_raw = "".join(thinking_buf)
                logger.debug(f"Raw thinking content: {repr(thinking_raw[:100])}")
                
                # Apply minimal cleaning
                thinking_text = thinking_raw
                if '<details' in thinking_text or '<summary' in thinking_text:
                    thinking_text = self._clean_thinking_content(thinking_text)
                
                # Join raw answer content
                answer_raw = "".join(answer_buf) if answer_buf else ""
                logger.debug(f"Raw answer content: {repr(answer_raw[:100])}")
                
                # Apply minimal cleaning
                answer_text = answer_raw
                if '<details' in answer_text:
                    answer_text = self._clean_answer_content(answer_text)
                
                if thinking_text:
                    final_text = f"<think>{thinking_text}</think>{answer_text}"
                else:
                    final_text = answer_text
            else:
                # Process answer content only
                answer_raw = "".join(answer_buf)
                logger.debug(f"Raw answer content (no thinking): {repr(answer_raw[:100])}")
                
                final_text = answer_raw
                if '<details' in final_text:
                    final_text = self._clean_answer_content(final_text)

            logger.debug(f"Final text: {repr(final_text[:100])}")

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
            logger.error(f"Request error in non-stream: {e}")
            if 'cookie' in locals():
                await cookie_manager.mark_cookie_invalid(cookie)
            raise HTTPException(status_code=502, detail="Upstream connection error")
        except Exception as e:
            logger.error(f"Unexpected error in non-stream: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            if client:
                await client.aclose()

    # FastAPI entry point
    async def handle_chat_completion(self, request: ChatCompletionRequest):
        """Main handler for chat completion requests"""
        stream = bool(request.stream) if request.stream is not None else settings.DEFAULT_STREAM
        if stream:
            return StreamingResponse(
                self.stream_proxy_response(request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            return await self.non_stream_proxy_response(request)