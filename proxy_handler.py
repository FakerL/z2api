"""
Proxy handler for Z.AI API requests
"""

import json
import logging
import re
import time
from typing import AsyncGenerator, Dict, Any
import httpx
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
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def handle_chat_completion(self, request: ChatCompletionRequest):
        """Handle chat completion request by streaming or aggregating."""
        is_streaming = (
            request.stream if request.stream is not None else settings.DEFAULT_STREAM
        )

        response_generator = self._stream_proxy_logic(request)

        if is_streaming:
            return StreamingResponse(
                response_generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            full_content = ""
            final_id = f"chatcmpl-{time.time()}"
            async for chunk_data in response_generator:
                if chunk_data.startswith("data: ") and not chunk_data.startswith("data: [DONE]"):
                    try:
                        chunk_json_str = chunk_data[6:].strip()
                        if not chunk_json_str:
                            continue
                        chunk_json = json.loads(chunk_json_str)
                        final_id = chunk_json.get("id", final_id)
                        delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                        if delta and delta.get("content"):
                            full_content += delta["content"]
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode JSON from stream chunk: {chunk_data}")
                        continue
            
            return ChatCompletionResponse(
                id=final_id,
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

    async def _stream_proxy_logic(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """
        Core logic for streaming from Z.AI and transforming to OpenAI format.
        """
        cookie = await cookie_manager.get_next_cookie()
        if not cookie:
            error_chunk = {"error": {"message": "No available cookies", "type": "server_error", "code": 503}}
            yield f"data: {json.dumps(error_chunk)}\n\n"
            return
        
        target_model = settings.UPSTREAM_MODEL
        import uuid
        
        # KEY FIX: Add "enable_thinking": True to the features object to force the thinking phase.
        request_data = {
            "stream": True,
            "model": target_model,
            "messages": request.model_dump(exclude_none=True)["messages"],
            "features": {
                "image_generation": False,
                "code_interpreter": False,
                "web_search": False,
                "auto_web_search": False,
                "enable_thinking": True,  # This is the critical flag.
            },
            "id": str(uuid.uuid4()),
            "model_item": {"id": target_model, "name": "GLM-4.5", "owned_by": "openai"},
            "variables": {
                "{{USER_NAME}}": "User",
                "{{USER_LOCATION}}": "Unknown",
                "{{CURRENT_DATETIME}}": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        headers = {
            "Content-Type": "application/json", "Authorization": f"Bearer {cookie}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
            "Accept": "application/json, text/event-stream", "Accept-Language": "zh-CN",
            "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"', "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"', "x-fe-version": "prod-fe-1.0.53", "Origin": "https://chat.z.ai",
            "Referer": "https://chat.z.ai/",
        }

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        buffer = ""

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=300.0), limits=httpx.Limits(max_connections=100, max_keepalive_connections=20), http2=True
        ) as client:
            try:
                async with client.stream("POST", settings.UPSTREAM_URL, json=request_data, headers=headers) as response:
                    if response.status_code != 200:
                        await cookie_manager.mark_cookie_failed(cookie)
                        error_text = await response.aread()
                        error_detail = error_text.decode('utf-8', errors='ignore')
                        error_code = response.status_code
                        error_type = "auth_error" if error_code == 401 else "server_error"
                        error_chunk = {"error": {"message": f"Upstream error ({error_code}): {error_detail}", "type": error_type, "code": error_code}}
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        return

                    await cookie_manager.mark_cookie_success(cookie)

                    async for chunk in response.aiter_text():
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line.startswith("data:"):
                                continue
                            
                            payload_str = line[6:].strip()
                            if payload_str == "[DONE]":
                                continue

                            try:
                                parsed = json.loads(payload_str)
                                data = parsed.get("data", {})
                                delta_content = data.get("delta_content", "")
                                phase = data.get("phase", "").strip()

                                if not delta_content:
                                    continue

                                output_content = ""
                                if settings.SHOW_THINK_TAGS:
                                    temp_content = re.sub(r'<details[^>]*>\s*<summary>.*?</summary>', '<think>', delta_content, flags=re.DOTALL)
                                    output_content = temp_content.replace('</details>', '</think>')
                                else:
                                    if phase == "thinking":
                                        output_content = ""
                                    elif phase == "answer":
                                        if '</details>' in delta_content:
                                            output_content = delta_content.split('</details>', 1)[1]
                                        else:
                                            output_content = delta_content
                                
                                if output_content:
                                    yield f"data: {json.dumps(self._create_openai_chunk(completion_id, request.model, output_content))}\n\n"

                            except json.JSONDecodeError:
                                logger.debug(f"Skipping non-JSON data line: {line}")
                                continue

                final_chunk = self._create_openai_chunk(completion_id, request.model, "", is_final=True)
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            except httpx.RequestError as e:
                logger.error(f"Streaming request error: {e}")
                await cookie_manager.mark_cookie_failed(cookie)
                error_chunk = {"error": {"message": f"Upstream service unavailable: {str(e)}", "type": "server_error"}}
                yield f"data: {json.dumps(error_chunk)}\n\n"

    def _create_openai_chunk(self, completion_id: str, model: str, content: str, is_final: bool = False) -> Dict[str, Any]:
        """A helper to create a standard OpenAI stream chunk."""
        chunk = {
            "id": completion_id, "object": "chat.completion.chunk", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": None}]
        }
        if is_final:
            chunk["choices"][0]["finish_reason"] = "stop"
        else:
            chunk["choices"][0]["delta"] = {"content": content}
        return chunk