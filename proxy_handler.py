"""
Proxy handler for Z.AI API requests
"""
import json, logging, re, time, uuid
from typing import AsyncGenerator, Dict, Any, Tuple, List

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from cookie_manager import cookie_manager
from models import ChatCompletionRequest, ChatCompletionResponse

logger = logging.getLogger(__name__)


class ProxyHandler:
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=300.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            http2=True,
        )

    async def aclose(self):
        if not self.client.is_closed:
            await self.client.aclose()
    
    def _clean_thinking_content(self, text: str) -> str:
        """
        Aggressively cleans raw thinking content strings.
        Removes tool calls, HTML-like tags, and other metadata.
        """
        if not text:
            return ""
        # Remove tool call blocks first
        cleaned_text = re.sub(r'<glm_block.*?</glm_block>', '', text, flags=re.DOTALL)
        # Remove all other HTML-like tags
        cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
        # Remove specific metadata patterns
        cleaned_text = re.sub(r'true" duration="\d+">\s*Thought for \d+ seconds', '', cleaned_text)
        # Remove leading markdown quote symbols
        cleaned_text = re.sub(r'^\s*>\s*', '', cleaned_text, flags=re.MULTILINE)
        # Remove "Thinking..." headers
        cleaned_text = cleaned_text.replace("Thinking…", "")
        # Final strip to clean up residual whitespace
        return cleaned_text.strip()

    def _clean_answer_content(self, text: str) -> str:
        """
        Cleans only <glm_block> tags from the final answer content,
        preserving other potential markdown or HTML formatting.
        """
        if not text:
            return ""
        # Remove only tool call blocks
        cleaned_text = re.sub(r'<glm_block.*?</glm_block>', '', text, flags=re.DOTALL)
        return cleaned_text.strip()

    def _serialize_msgs(self, msgs) -> list:
        """Converts message objects to a list of dictionaries."""
        out = []
        for m in msgs:
            if hasattr(m, "dict"): out.append(m.dict())
            elif hasattr(m, "model_dump"): out.append(m.model_dump())
            elif isinstance(m, dict): out.append(m)
            else: out.append({"role": getattr(m, "role", "user"), "content": getattr(m, "content", str(m))})
        return out

    async def _prep_upstream(self, req: ChatCompletionRequest) -> Tuple[Dict[str, Any], Dict[str, str], str]:
        """Prepares the request body, headers, and cookie for the upstream API."""
        ck = await cookie_manager.get_next_cookie()
        if not ck: raise HTTPException(503, "No available cookies")
        model = settings.UPSTREAM_MODEL if req.model == settings.MODEL_NAME else req.model
        body = { "stream": True, "model": model, "messages": self._serialize_msgs(req.messages), "background_tasks": {"title_generation": True, "tags_generation": True}, "chat_id": str(uuid.uuid4()), "features": {"image_generation": False, "code_interpreter": False, "web_search": False, "auto_web_search": False, "enable_thinking": True,}, "id": str(uuid.uuid4()), "mcp_servers": ["deep-web-search"], "model_item": {"id": model, "name": "GLM-4.5", "owned_by": "openai"}, "params": {}, "tool_servers": [], "variables": {"{{USER_NAME}}": "User", "{{USER_LOCATION}}": "Unknown", "{{CURRENT_DATETIME}}": time.strftime("%Y-%m-%d %H:%M:%S"),},}
        headers = { "Content-Type": "application/json", "Authorization": f"Bearer {ck}", "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"), "Accept": "application/json, text/event-stream", "Accept-Language": "zh-CN", "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"', "sec-ch-ua-mobile": "?0", "sec-ch-ua-platform": '"macOS"', "x-fe-version": "prod-fe-1.0.53", "Origin": "https://chat.z.ai", "Referer": "https://chat.z.ai/",}
        return body, headers, ck
        
    async def stream_proxy_response(self, req: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        ck = None
        try:
            body, headers, ck = await self._prep_upstream(req)
            comp_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"; think_open = False; phase_cur = None

            async def yield_content(content_type: str, text: str):
                nonlocal think_open
                if not text: return
                
                # Apply cleaning based on content type
                cleaned_text = ""
                if content_type == "thinking":
                    cleaned_text = self._clean_thinking_content(text)
                elif content_type == "answer":
                    cleaned_text = self._clean_answer_content(text)

                if not cleaned_text: return

                if content_type == "thinking" and settings.SHOW_THINK_TAGS:
                    if not think_open:
                        yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '<think>'}, 'finish_reason': None}]})}\n\n"
                        think_open = True
                    
                    yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': cleaned_text}, 'finish_reason': None}]})}\n\n"
                
                elif content_type == "answer":
                    if think_open:
                        yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '</think>'}, 'finish_reason': None}]})}\n\n"
                        think_open = False
                    
                    yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': cleaned_text}, 'finish_reason': None}]})}\n\n"

            async with self.client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_failed(ck); err_body = await resp.aread()
                    err_msg = f"Error: {resp.status_code} - {err_body.decode(errors='ignore')}"
                    err = {"id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": req.model, "choices": [{"index": 0, "delta": {"content": err_msg}, "finish_reason": "stop"}],}
                    yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"; return
                await cookie_manager.mark_cookie_success(ck)
                
                async for raw in resp.aiter_text():
                    for line in raw.strip().split('\n'):
                        line = line.strip()
                        if not line.startswith('data: '): continue
                        
                        payload_str = line[6:]
                        if payload_str == '[DONE]':
                            if think_open: yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '</think>'}, 'finish_reason': None}]})}\n\n"
                            yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"; yield "data: [DONE]\n\n"; return
                        try:
                            dat = json.loads(payload_str).get("data", {})
                        except (json.JSONDecodeError, AttributeError): continue
                        
                        new_phase = dat.get("phase")
                        if new_phase: phase_cur = new_phase
                        if not phase_cur: continue
                        
                        content = dat.get("delta_content") or dat.get("edit_content")
                        if not content: continue

                        match = re.search(r'(.*</details>)(.*)', content, flags=re.DOTALL)
                        if match:
                            thinking_part, answer_part = match.groups()
                            async for item in yield_content("thinking", thinking_part): yield item
                            async for item in yield_content("answer", answer_part): yield item
                        else:
                             async for item in yield_content(phase_cur, content): yield item
        except Exception:
            logger.exception("Stream error"); raise

    async def non_stream_proxy_response(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        ck = None
        try:
            body, headers, ck = await self._prep_upstream(req)
            raw_thinking_parts = []; raw_answer_parts = []; phase_cur = None
            async with self.client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_failed(ck); error_detail = await resp.text()
                    raise HTTPException(resp.status_code, f"Upstream error: {error_detail}")
                await cookie_manager.mark_cookie_success(ck)
                
                async for raw in resp.aiter_text():
                    for line in raw.strip().split('\n'):
                        line = line.strip()
                        if not line.startswith('data: '): continue
                        
                        payload_str = line[6:]
                        if payload_str == '[DONE]': break
                        try:
                            dat = json.loads(payload_str).get("data", {})
                        except (json.JSONDecodeError, AttributeError): continue
                        
                        new_phase = dat.get("phase")
                        if new_phase: phase_cur = new_phase
                        if not phase_cur: continue

                        content = dat.get("delta_content") or dat.get("edit_content")
                        if not content: continue
                        
                        match = re.search(r'(.*</details>)(.*)', content, flags=re.DOTALL)
                        if match:
                            thinking_part, answer_part = match.groups()
                            raw_thinking_parts.append(thinking_part)
                            raw_answer_parts.append(answer_part)
                        else:
                            if phase_cur == "thinking":
                                raw_thinking_parts.append(content)
                            elif phase_cur == "answer":
                                raw_answer_parts.append(content)
                    else: continue
                    break

            # Clean the final answer text, removing only <glm_block> tags.
            cleaned_ans_text = self._clean_answer_content(''.join(raw_answer_parts))
            final_content = cleaned_ans_text
            
            if settings.SHOW_THINK_TAGS and raw_thinking_parts:
                # Aggressively clean the thinking part.
                cleaned_think_text = self._clean_thinking_content(''.join(raw_thinking_parts))
                if cleaned_think_text:
                    final_content = f"<think>{cleaned_think_text}</think>{cleaned_ans_text}"

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}", created=int(time.time()), model=req.model,
                choices=[{"index": 0, "message": {"role": "assistant", "content": final_content}, "finish_reason": "stop"}],
            )
        except Exception:
            logger.exception("Non-stream processing failed"); raise

    async def handle_chat_completion(self, req: ChatCompletionRequest):
        """Determines whether to stream or not and handles the request."""
        stream = bool(req.stream) if req.stream is not None else settings.DEFAULT_STREAM
        if stream:
            return StreamingResponse(self.stream_proxy_response(req), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
        return await self.non_stream_proxy_response(req)