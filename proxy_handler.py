"""
Proxy handler for Z.AI API requests
"""
import json, logging, re, time, uuid
from typing import AsyncGenerator, Dict, Any, Tuple

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

    # --- Text utilities (REVISED _clean_answer_from_edit) ---
    def _clean_thinking(self, s: str) -> str:
        if not s: return ""
        s = re.sub(r'<details[^>]*>.*?</details>', '', s, flags=re.DOTALL)
        s = re.sub(r'<summary[^>]*>.*?</summary>', '', s, flags=re.DOTALL)
        s = re.sub(r'<[^>]+>', '', s)
        s = re.sub(r'^\s*>\s*', '', s, flags=re.MULTILINE)
        return s

    def _clean_answer(self, s: str, from_edit_content: bool = False) -> str:
        """
        Cleans the answer string.
        If from_edit_content is True, it extracts only the content after the last </details> tag.
        """
        if not s: return ""
        
        if from_edit_content:
            # Find the position of the last </details> tag
            last_details_pos = s.rfind('</details>')
            if last_details_pos != -1:
                # Extract everything after the tag
                s = s[last_details_pos + len('</details>'):]
        
        # General cleanup for any remaining <details> blocks (just in case)
        s = re.sub(r"<details[^>]*>.*?</details>", "", s, flags=re.DOTALL)
        return s.lstrip() # Use lstrip to remove leading whitespace like '\n' but keep internal space

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
        
    # ---------- stream ----------
    async def stream_proxy_response(self, req: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        ck = None
        try:
            body, headers, ck = await self._prep_upstream(req)
            comp_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            think_open = False
            phase_cur = None

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
                        
                        # FIX: Differentiate content source
                        is_edit = "edit_content" in dat
                        content = dat.get("delta_content", "") or dat.get("edit_content", "")
                        new_phase = dat.get("phase")

                        is_transition = new_phase and new_phase != phase_cur
                        if is_transition:
                            if phase_cur == "thinking" and new_phase == "answer" and think_open and settings.SHOW_THINK_TAGS:
                                close_payload = {'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '</think>'}, 'finish_reason': None}]}
                                yield f"data: {json.dumps(close_payload)}\n\n"
                                think_open = False
                            phase_cur = new_phase
                        
                        current_content_phase = phase_cur or new_phase
                        
                        text_to_yield = ""
                        if current_content_phase == "thinking":
                            if content and settings.SHOW_THINK_TAGS:
                                if not think_open:
                                    open_payload = {'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '<think>'}, 'finish_reason': None}]}
                                    yield f"data: {json.dumps(open_payload)}\n\n"
                                    think_open = True
                                text_to_yield = self._clean_thinking(content)
                        elif current_content_phase == "answer":
                            # FIX: Use the new cleaning logic
                            text_to_yield = self._clean_answer(content, from_edit_content=is_edit)

                        if text_to_yield:
                            content_payload = {"id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": req.model, "choices": [{"index": 0, "delta": {"content": text_to_yield}, "finish_reason": None}],}
                            yield f"data: {json.dumps(content_payload)}\n\n"
        except Exception:
            logger.exception("Stream error"); raise

    # ---------- non-stream ----------
    async def non_stream_proxy_response(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        ck = None
        try:
            body, headers, ck = await self._prep_upstream(req)
            full_content = []
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
                        
                        is_edit = "edit_content" in dat
                        content = dat.get("delta_content") or dat.get("edit_content")
                        new_phase = dat.get("phase")
                        
                        if new_phase: phase_cur = new_phase
                        if content and phase_cur:
                            # Store the content along with its source (is_edit)
                            full_content.append((phase_cur, content, is_edit))
                    else: continue
                    break
            
            think_buf = []
            answer_buf = []
            for phase, content, is_edit in full_content:
                if phase == "thinking":
                    think_buf.append(self._clean_thinking(content))
                elif phase == "answer":
                    # FIX: Use the new cleaning logic
                    answer_buf.append(self._clean_answer(content, from_edit_content=is_edit))
            
            ans_text = ''.join(answer_buf)
            final_content = ans_text

            if settings.SHOW_THINK_TAGS and think_buf:
                think_text = ''.join(think_buf).strip()
                if think_text:
                    # No longer need to manually add newline, as .lstrip() in _clean_answer handles it
                    final_content = f"<think>{think_text}</think>{ans_text}"

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