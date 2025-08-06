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
            content = re.sub(r'<details[^>]*>.*?</details>', '', content, flags=re.DOTALL)
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
                think_start = content.find('<think>')
                if think_start != -1:
                    answer_match = re.search(r'\n\s*[A-Z0-9]', content[think_start:])
                    if answer_match:
                        insert_pos = think_start + answer_match.start()
                        content = content[:insert_pos] + '</think>\n' + content[insert_pos:]
                    else:
                        content += '</think>'

        return content.strip()
    
    def transform_delta_content(self, content: str) -> str:
        """Transform delta content for streaming"""
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
    
    # Build request data based on the actual Z.AI API format
    import uuid
    from datetime import datetime

    current_time = datetime.now()
    
    # Generate unique IDs for the request
    chat_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    
    # Transform messages to include message_id
    messages_with_ids = []
    for msg in request.model_dump()["messages"]:
        message_with_id = {
            **msg,
            "message_id": str(uuid.uuid4())  # Add message_id to each message
        }
        messages_with_ids.append(message_with_id)
    
    request_data = {
        "stream": True,
        "model": target_model,
        "messages": messages_with_ids,  # Use messages with IDs
        "chat_id": chat_id,  # Add chat_id
        "id": request_id,    # Add request ID
        "params": {},
        "tool_servers": [],
        "features": {
            "image_generation": False,
            "code_interpreter": False,
            "web_search": False,
            "auto_web_search": False,
            "preview_mode": True,
            "flags": [],
            "features": [
                {"type": "mcp", "server": "vibe-coding", "status": "hidden"},
                {"type": "mcp", "server": "ppt-maker", "status": "hidden"},
                {"type": "mcp", "server": "image-search", "status": "hidden"}
            ],
            "enable_thinking": True
        },
        "variables": {
            "{{USER_NAME}}": "User",
            "{{USER_LOCATION}}": "Unknown",
            "{{CURRENT_DATETIME}}": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "{{CURRENT_DATE}}": current_time.strftime("%Y-%m-%d"),
            "{{CURRENT_TIME}}": current_time.strftime("%H:%M:%S"),
            "{{CURRENT_WEEKDAY}}": current_time.strftime("%A"),
            "{{CURRENT_TIMEZONE}}": "Asia/Taipei",
            "{{USER_LANGUAGE}}": "zh-CN"
        },
        "model_item": {
            "id": target_model,
            "name": "GLM-4.5",
            "owned_by": "openai",
            "openai": {
                "id": target_model,
                "name": target_model,
                "owned_by": "openai",
                "openai": {"id": target_model},
                "urlIdx": 1
            },
            "urlIdx": 1,
            "info": {
                "id": target_model,
                "user_id": "7080a6c5-5fcc-4ea4-a85f-3b3fac905cf2",
                "base_model_id": None,
                "name": "GLM-4.5",
                "params": {
                    "top_p": 0.95,
                    "temperature": 0.6,
                    "max_tokens": 80000
                },
                "meta": {
                    "profile_image_url": "/static/favicon.png",
                    "description": "Most advanced model, proficient in coding and tool use",
                    "capabilities": {
                        "vision": False,
                        "citations": False,
                        "preview_mode": False,
                        "web_search": False,
                        "language_detection": False,
                        "restore_n_source": False,
                        "mcp": True,
                        "file_qa": True,
                        "returnFc": True,
                        "returnThink": True,
                        "think": True
                    },
                    "mcpServerIds": ["deep-web-search", "ppt-maker", "image-search", "vibe-coding"]
                }
            }
        }
    }

    logger.debug(f"Sending request data: {json.dumps(request_data, indent=2)}")
    
    # Use the exact headers from your curl request
    headers = {
        "Accept": "*/*",
        "Accept-Language": "zh-CN",
        "Authorization": f"Bearer {cookie}",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "Cookie": f"token={cookie}",
        "Origin": "https://chat.z.ai",
        "Referer": "https://chat.z.ai/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Mobile Safari/537.36",
        "X-FE-Version": "prod-fe-1.0.57",
        "sec-ch-ua": '"Chromium";v="137", "Not/A)Brand";v="24"',
        "sec-ch-ua-mobile": "?1",
        "sec-ch-ua-platform": '"Android"'
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
            logger.error(f"Upstream error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Upstream error: {response.text}")
        
        await cookie_manager.mark_cookie_success(cookie)
        return {"response": response, "cookie": cookie}
        
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        await cookie_manager.mark_cookie_failed(cookie)
        raise HTTPException(status_code=503, detail="Upstream service unavailable")
    
    async def process_streaming_response_real_time(self, response: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
        """Process streaming response in real time - truly streaming"""
        buffer = ""
        
        async for chunk in response.aiter_text():
            if not chunk:
                continue
                
            buffer += chunk
            lines = buffer.split('\n')
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

                    # Check for errors first
                    if parsed.get("error") or (parsed.get("data", {}).get("error")):
                        error_detail = (parsed.get("error", {}).get("detail") or
                                      parsed.get("data", {}).get("error", {}).get("detail") or
                                      "Unknown error from upstream")
                        logger.error(f"Upstream error: {error_detail}")
                        raise HTTPException(status_code=400, detail=f"Upstream error: {error_detail}")

                    # Transform the response immediately
                    if parsed.get("data"):
                        # Remove unwanted fields
                        parsed["data"].pop("edit_index", None)
                        parsed["data"].pop("edit_content", None)

                        # Transform delta_content immediately for streaming
                        delta_content = parsed["data"].get("delta_content", "")
                        if delta_content and settings.SHOW_THINK_TAGS:
                            transformed = self.transform_delta_content(delta_content)
                            parsed["data"]["delta_content"] = transformed.lstrip()

                    # Yield immediately - this is true streaming!
                    yield parsed

                except json.JSONDecodeError as e:
                    logger.debug(f"JSON decode error for line: {line[:100]}... Error: {e}")
                    continue  # Skip non-JSON lines
    
    async def handle_chat_completion(self, request: ChatCompletionRequest):
        """Handle chat completion request"""
        proxy_result = await self.proxy_request(request)
        response = proxy_result["response"]

        # Determine final streaming mode
        is_streaming = request.stream if request.stream is not None else settings.DEFAULT_STREAM

        if is_streaming:
            return StreamingResponse(
                self.stream_response_real_time(response, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            return await self.non_stream_response(response, request.model)

    async def stream_response_real_time(self, response: httpx.Response, model: str) -> AsyncGenerator[str, None]:
        """Generate truly real-time streaming response in OpenAI format"""
        import uuid
        import time
        
        # Generate a unique completion ID
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        
        try:
            # Process each chunk immediately as it arrives - true streaming!
            async for parsed in self.process_streaming_response_real_time(response):
                try:
                    data = parsed.get("data", {})
                    delta_content = data.get("delta_content", "")
                    phase = data.get("phase", "")
                    
                    # For SHOW_THINK_TAGS=false, filter out non-answer content
                    if not settings.SHOW_THINK_TAGS and phase != "answer" and delta_content:
                        logger.debug(f"Skipping content in {phase} phase (SHOW_THINK_TAGS=false)")
                        continue
                    
                    # Send content immediately if available
                    if delta_content:
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
                        
                        chunk_json = json.dumps(openai_chunk)
                        yield f"data: {chunk_json}\n\n"
                        logger.debug(f"Sent chunk: {chunk_json[:100]}...")
                
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
        """Generate non-streaming response by collecting all chunks"""
        chunks = []
        
        # For non-streaming, we still collect all chunks first
        async for parsed in self.process_streaming_response_real_time(response):
            chunks.append(parsed)
            logger.debug(f"Collected chunk: {parsed.get('data', {}).get('delta_content', '')[:50]}...")

        if not chunks:
            raise HTTPException(status_code=500, detail="No response from upstream")

        logger.info(f"Total chunks collected: {len(chunks)}")

        # Aggregate content based on SHOW_THINK_TAGS setting
        if settings.SHOW_THINK_TAGS:
            full_content = "".join(
                chunk.get("data", {}).get("delta_content", "") for chunk in chunks
            )
        else:
            full_content = "".join(
                chunk.get("data", {}).get("delta_content", "")
                for chunk in chunks
                if chunk.get("data", {}).get("phase") == "answer"
            )

        logger.info(f"Aggregated content length: {len(full_content)}")

        # Apply final content transformation for non-streaming
        transformed_content = self.transform_content(full_content)

        logger.info(f"Final transformed content length: {len(transformed_content)}")

        # Create OpenAI-compatible response
        return ChatCompletionResponse(
            id=chunks[0].get("data", {}).get("id", "chatcmpl-unknown") if chunks else "chatcmpl-unknown",
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