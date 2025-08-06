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
from models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse

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

        # Optionally remove thinking content based on configuration
        if not settings.SHOW_THINK_TAGS:
            logger.debug("Removing thinking content from response")
            original_length = len(content)

            # Remove <details> blocks (thinking content) - handle both closed and unclosed tags
            # First try to remove complete <details>...</details> blocks
            content = re.sub(r'<details[^>]*>.*?</details>', '', content, flags=re.DOTALL)

            # Then remove any remaining <details> opening tags and everything after them until we hit answer content
            # Look for pattern: <details...><summary>...</summary>...content... and remove the thinking part
            content = re.sub(r'<details[^>]*>.*?(?=\s*[A-Z]|\s*\d|\s*$)', '', content, flags=re.DOTALL)

            content = content.strip()

            logger.debug(f"Content length after removing thinking content: {original_length} -> {len(content)}")
        else:
            logger.debug("Keeping thinking content, converting to <think> tags")

            # Replace <details> with <think>
            content = re.sub(r'<details[^>]*>', '<think>', content)
            content = content.replace('</details>', '</think>')

            # Remove <summary> tags and their content
            content = re.sub(r'<summary>.*?</summary>', '', content, flags=re.DOTALL)

            # If there's no closing </think>, add it at the end of thinking content
            if '<think>' in content and '</think>' not in content:
                # Find where thinking ends and answer begins
                think_start = content.find('<think>')
                if think_start != -1:
                    # Look for the start of the actual answer (usually starts with a capital letter or number)
                    answer_match = re.search(r'\n\s*[A-Z0-9]', content[think_start:])
                    if answer_match:
                        insert_pos = think_start + answer_match.start()
                        content = content[:insert_pos] + '</think>\n' + content[insert_pos:]
                    else:
                        content += '</think>'

        return content.strip()
    
    def transform_delta_content(self, content: str) -> str:
        """Transform delta content for streaming (simpler version for chunk processing)"""
        if not content:
            return content
            
        # Convert <details> to <think> and remove summary tags
        content = re.sub(r'<details[^>]*>', '<think>', content)
        content = content.replace('</details>', '</think>')
        content = re.sub(r'<summary>.*?</summary>', '', content, flags=re.DOTALL)
        
        return content
    
    async def proxy_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Proxy request to Z.AI API"""
        cookie = await cookie_manager.get_next_cookie()
        if not cookie:
            raise HTTPException(status_code=503, detail="No available cookies")
        
        # Transform model name
        target_model = settings.UPSTREAM_MODEL if request.model == settings.MODEL_NAME else request.model
        
        # Determine if this should be a streaming response
        is_streaming = request.stream if request.stream is not None else settings.DEFAULT_STREAM

        # Validate parameter compatibility
        if is_streaming and not settings.SHOW_THINK_TAGS:
            logger.warning("SHOW_THINK_TAGS=false is ignored for streaming responses")

        # Prepare request data
        request_data = request.model_dump(exclude_none=True)
        request_data["model"] = target_model

        # Build request data based on actual Z.AI format from zai-messages.md
        import uuid

        request_data = {
            "stream": True,  # Always request streaming from Z.AI for processing
            "model": target_model,
            "messages": request_data["messages"],
            "background_tasks": {
                "title_generation": True,
                "tags_generation": True
            },
            "chat_id": str(uuid.uuid4()),
            "features": {
                "image_generation": False,
                "code_interpreter": False,
                "web_search": False,
                "auto_web_search": False
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
            "Referer": "https://chat.z.ai/c/069723d5-060b-404f-992c-4705f1554c4c"
        }
        
        try:
            response = await self.client.post(
                settings.UPSTREAM_URL,
                json=request_data,
                headers=headers
            )
            
            if response.status_code == 401:
                await cookie_manager.mark_cookie_failed(cookie)
                raise HTTPException(status_code=401, detail="Invalid authentication")
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"Upstream error: {response.text}")
            
            await cookie_manager.mark_cookie_success(cookie)
            return {"response": response, "cookie": cookie}
            
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            await cookie_manager.mark_cookie_failed(cookie)
            raise HTTPException(status_code=503, detail="Upstream service unavailable")
    
    async def process_and_collect_chunks(self, response: httpx.Response) -> list:
        """Collect and process all chunks similar to Deno version"""
        chunks = []
        buffer = ""
        
        async for chunk in response.aiter_text():
            buffer += chunk
            lines = buffer.split('\n')
            buffer = lines[-1]  # Keep incomplete line in buffer
            
            for line in lines[:-1]:
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                
                payload = line[6:].strip()
                if payload == "[DONE]":
                    return chunks
                
                try:
                    parsed = json.loads(payload)

                    # Check for errors first
                    if parsed.get("error") or (parsed.get("data", {}).get("error")):
                        error_detail = (parsed.get("error", {}).get("detail") or
                                      parsed.get("data", {}).get("error", {}).get("detail") or
                                      "Unknown error from upstream")
                        logger.error(f"Upstream error: {error_detail}")
                        raise HTTPException(status_code=400, detail=f"Upstream error: {error_detail}")

                    # Transform the response similar to Deno version
                    if parsed.get("data"):
                        # Remove unwanted fields
                        parsed["data"].pop("edit_index", None)
                        parsed["data"].pop("edit_content", None)

                        # Transform delta_content
                        delta_content = parsed["data"].get("delta_content", "")
                        if delta_content:
                            transformed = self.transform_delta_content(delta_content)
                            parsed["data"]["delta_content"] = transformed.lstrip()

                    chunks.append(parsed)

                except json.JSONDecodeError:
                    continue  # Skip non-JSON lines
        
        return chunks
    
    async def handle_chat_completion(self, request: ChatCompletionRequest):
        """Handle chat completion request"""
        proxy_result = await self.proxy_request(request)
        response = proxy_result["response"]

        # Determine final streaming mode
        is_streaming = request.stream if request.stream is not None else settings.DEFAULT_STREAM

        if is_streaming:
            return StreamingResponse(
                self.stream_response(response, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            return await self.non_stream_response(response, request.model)

    async def stream_response(self, response: httpx.Response, model: str) -> AsyncGenerator[str, None]:
        """Generate streaming response in OpenAI format - collect chunks first then stream"""
        import uuid
        import time
        
        # Generate a unique completion ID
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        
        try:
            # Collect all chunks first (similar to Deno approach)
            chunks = await self.process_and_collect_chunks(response)
            
            if not chunks:
                raise HTTPException(status_code=500, detail="No response from upstream")
            
            # Stream back the transformed chunks
            for parsed in chunks:
                try:
                    data = parsed.get("data", {})
                    delta_content = data.get("delta_content", "")
                    
                    # For SHOW_THINK_TAGS=false, filter out thinking content
                    if not settings.SHOW_THINK_TAGS:
                        phase = data.get("phase", "")
                        if phase != "answer" and delta_content:
                            continue  # Skip non-answer content
                    
                    # Create OpenAI-compatible streaming chunk
                    if delta_content:  # Only send chunks with content
                        openai_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": delta_content
                                },
                                "finish_reason": None
                            }]
                        }
                        
                        yield f"data: {json.dumps(openai_chunk)}\n\n"
                
                except Exception as e:
                    logger.error(f"Error processing streaming chunk: {e}")
                    continue
            
            # Send final completion chunk
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk", 
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            # Send error in OpenAI format
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    async def non_stream_response(self, response: httpx.Response, model: str) -> ChatCompletionResponse:
        """Generate non-streaming response"""
        # Collect all chunks
        chunks = await self.process_and_collect_chunks(response)

        if not chunks:
            raise HTTPException(status_code=500, detail="No response from upstream")

        logger.info(f"Total chunks received: {len(chunks)}")

        # Aggregate content based on SHOW_THINK_TAGS setting
        if settings.SHOW_THINK_TAGS:
            # Include all content
            full_content = "".join(
                chunk.get("data", {}).get("delta_content", "") for chunk in chunks
            )
        else:
            # Only include answer phase content
            full_content = "".join(
                chunk.get("data", {}).get("delta_content", "")
                for chunk in chunks
                if chunk.get("data", {}).get("phase") == "answer"
            )

        logger.info(f"Aggregated content length: {len(full_content)}")

        # Apply final content transformation (for non-streaming, we can do more complex transformations)
        transformed_content = self.transform_content(full_content)

        logger.info(f"Transformed content length: {len(transformed_content)}")

        # Create OpenAI-compatible response
        return ChatCompletionResponse(
            id=chunks[0].get("data", {}).get("id", "chatcmpl-unknown"),
            created=int(time.time()),
            model=model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": transformed_content
                },
                "finish_reason": "stop"
            }]
        )