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

    # --- Text utilities ---
    # These are now simplified, as the main logic will handle parsing.
    def _clean_html_and_quotes(self, s: str) -> str:
        if not s: return ""
        s = re.sub(r'<details[^>]*>.*?</details>', '', s, flags=re.DOTALL)
        s = re.sub(r'<summary[^>]*>.*?</summary>', '', s, flags=re.DOTALL)
        s = re.sub(r'<[^>]+>', '', s)
        s = re.sub(r'^\s*>\s*', '', s, flags=re.MULTILINE)
        return s

    # ... Other methods like _serialize_msgs, _prep_upstream remain the same ...
    def _serialize_msgs(self, msgs) -> list:
        out = []
        for m in msgs:
            if hasattr(m, "dict"): out.append(m.dict())
            elif hasattr(m, "model_dump"): out.append(m.model_dump())
            elif isinstance(m, dict): out.append(m)
            else: out.append({"role": getattr(m, "role", "user"), "content": getattr(m, "content", str(m))})
        return out
    async def _prep_upstream(self, req: ChatCompletionRequest) -> Tuple[Dict[str, Any], Dict[str, str], str]:
        ck = await cookie_manager.get_next_cookie()
        if not ck: raise HTTPException(503, "No available cookies")
        model = settings.UPSTREAM_MODEL if req.model == settings.MODEL_NAME else req.model
        body = { "stream": True, "model": model, "messages": self._serialize_msgs(req.messages), "background_tasks": {"title_generation": True, "tags_generation": True}, "chat_id": str(uuid.uuid4()), "features": {"image_generation": False, "code_interpreter": False, "web_search": False, "auto_web_search": False, "enable_thinking": True,}, "id": str(uuid.uuid4()), "mcp_servers": ["deep-web-search"], "model_item": {"id": model, "name": "GLM-4.5", "owned_by": "openai"}, "params": {}, "tool_servers": [], "variables": {"{{USER_NAME}}": "User", "{{USER_LOCATION}}": "Unknown", "{{CURRENT_DATETIME}}": time.strftime("%Y-%m-%d %H:%M:%S"),},}
        headers = { "Content-Type": "application/json", "Authorization": f"Bearer {ck}", "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"), "Accept": "application/json, text/event-stream", "Accept-Language": "zh-CN", "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"', "sec-ch-ua-mobile": "?0", "sec-ch-ua-platform": '"macOS"', "x-fe-version": "prod-fe-1.0.53", "Origin": "https://chat.z.ai", "Referer": "https://chat.z.ai/",}
        return body, headers, ck

    # --- NEW: Unified Chunk Parser ---
    def _parse_chunk(self, dat: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Parses a single data chunk from Z.AI and returns a list of (type, content) tuples.
        Type can be 'thinking' or 'answer'.
        This function is the core of the new logic, designed to be robust.
        """
        if not dat:
            return []

        content = dat.get("delta_content")
        edit_content = dat.get("edit_content")
        phase = dat.get("phase")

        # Case 1: Simple delta_content chunk
        if content and not edit_content:
            if phase in ["thinking", "answer"]:
                return [(phase, content)]
            return []

        # Case 2: Complex edit_content chunk (the source of most problems)
        if edit_content:
            # edit_content can contain both thinking and answer parts.
            # We split it by the last </details> tag.
            parts = edit_content.rsplit('</details>', 1)
            
            # This is a heuristic, but it's based on observed data.
            # If split results in 2 parts, the first is thinking, the second is answer.
            if len(parts) == 2:
                thinking_part, answer_part = parts
                return [("thinking", thinking_part + '</details>'), ("answer", answer_part)]
            else:
                # If no </details>, the whole thing is likely an answer or thinking chunk.
                if phase in ["thinking", "answer"]:
                    return [(phase, edit_content)]
        
        # Case 3: A chunk that only changes phase, without content.
        # We don't need to do anything, the state will be updated in the main loop.

        return []

    # ---------- stream (REBUILT) ----------
    async def stream_proxy_response(self, req: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        ck = None
        try:
            body, headers, ck = await self._prep_upstream(req)
            comp_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"; think_open = False; phase_cur = None

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
                        if not line or not line.startswith('data: '): continue
                        payload_str = line[6:]
                        if payload_str == '[DONE]':
                            if think_open: yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '</think>'}, 'finish_reason': None}]})}\n\n"
                            yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"; yield "data: [DONE]\n\n"; return
                        try:
                            dat = json.loads(payload_str).get("data", {})
                        except (json.JSONDecodeError, AttributeError): continue
                        
                        new_phase = dat.get("phase")
                        
                        # Handle phase transition side-effect
                        if new_phase and new_phase != phase_cur:
                            if phase_cur == "thinking" and think_open and settings.SHOW_THINK_TAGS:
                                yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '</think>'}, 'finish_reason': None}]})}\n\n"
                                think_open = False
                            phase_cur = new_phase
                        
                        # Use the unified parser
                        parsed_parts = self._parse_chunk(dat)
                        
                        for part_type, part_content in parsed_parts:
                            text_to_yield = ""
                            if part_type == "thinking":
                                if settings.SHOW_THINK_TAGS:
                                    if not think_open:
                                        yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '<think>'}, 'finish_reason': None}]})}\n\n"
                                        think_open = True
                                    text_to_yield = self._clean_html_and_quotes(part_content)
                            elif part_type == "answer":
                                # For answer, we pass it through raw to preserve all formatting.
                                text_to_yield = part_content
                            
                            # Always yield, even if empty, to preserve timing and newlines.
                            # The check `if part_content is not None` is implicit in the loop.
                            yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': text_to_yield}, 'finish_reason': None}]})}\n\n"

        except Exception:
            logger.exception("Stream error"); raise

    # ---------- non-stream (REBUILT) ----------
    async def non_stream_proxy_response(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        ck = None
        try:
            body, headers, ck = await self._prep_upstream(req)
            think_buf = []
            answer_buf = []
            phase_cur = None

            async with self.client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_failed(ck); error_detail = await resp.text()
                    raise HTTPException(resp.status_code, f"Upstream error: {error_detail}")
                await cookie_manager.mark_cookie_success(ck)
                
                async for raw in resp.aiter_text():
                    for line in raw.strip().split('\n'):
                        line = line.strip()
                        if not line or not line.startswith('data: '): continue
                        payload_str = line[6:]
                        if payload_str == '[DONE]': break
                        try:
                            dat = json.loads(payload_str).get("data", {})
                        except (json.JSONDecodeError, AttributeError): continue
                        
                        new_phase = dat.get("phase")
                        if new_phase:
                            phase_cur = new_phase
                        
                        # Use the unified parser
                        parsed_parts = self._parse_chunk(dat)

                        for part_type, part_content in parsed_parts:
                            if part_type == "thinking":
                                think_buf.append(part_content)
                            elif part_type == "answer":
                                answer_buf.append(part_content)
                    else: continue
                    break
            
            # Post-process collected buffers
            final_ans_text = ''.join(answer_buf)
            final_content = final_ans_text

            if settings.SHOW_THINK_TAGS and think_buf:
                # Clean the thinking part at the very end
                final_think_text = self._clean_html_and_quotes(''.join(think_buf)).strip()
                if final_think_text:
                    final_content = f"<think>{final_think_text}</think>{final_ans_text}"

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}", created=int(time.time()), model=req.model,
                choices=[{"index": 0, "message": {"role": "assistant", "content": final_content}, "finish_reason": "stop"}],
            )
        except Exception:
            logger.exception("Non-stream processing failed"); raise

    # ---------- FastAPI entry ----------
    async def handle_chat_completion(self, req: ChatCompletionRequest):
        stream = bool(req.stream) if req.stream is not None else settings.DEFAULT_STREAM
        if stream:
            return StreamingResponse(self.stream_proxy_response(req), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
        return await self.non_stream_proxy_response(req)