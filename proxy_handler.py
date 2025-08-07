"""
Proxy handler for Z.AI API requests
"""
import json, logging, re, time, uuid
from typing import AsyncGenerator, Dict, Any, Tuple

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

# 這些導入現在可以確定是正確的
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

    # --- Text utilities and other methods from the last robust version ---
    def _clean_thinking(self, s: str) -> str:
        if not s: return ""
        s = re.sub(r'<details[^>]*>.*?</details>', '', s, flags=re.DOTALL)
        s = re.sub(r'<summary[^>]*>.*?</summary>', '', s, flags=re.DOTALL)
        s = re.sub(r'<[^>]+>', '', s)
        s = re.sub(r'^\s*>\s*', '', s, flags=re.MULTILINE)
        return s
    def _clean_answer(self, s: str) -> str:
        if not s: return ""
        return re.sub(r"<details[^>]*>.*?</details>", "", s, flags=re.DOTALL)
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
                    # FIX: Using the correct method name `mark_cookie_failed`
                    await cookie_manager.mark_cookie_failed(ck)
                    err_body = await resp.aread(); err_msg = f"Error: {resp.status_code} - {err_body.decode(errors='ignore')}"
                    err = {"id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": req.model, "choices": [{"index": 0, "delta": {"content": err_msg}, "finish_reason": "stop"}],}
                    yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"; return
                
                # If the request was successful, the cookie is good.
                await cookie_manager.mark_cookie_success(ck)

                # The robust logic for handling chunks from the last attempt
                async for raw in resp.aiter_text():
                    for line in raw.strip().split('\n'):
                        line = line.strip()
                        if not line or not line.startswith('data: '): continue
                        payload_str = line[6:]
                        if payload_str == '[DONE]':
                            if think_open:
                                payload = {'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '</think>'}, 'finish_reason': None}]}
                                yield f"data: {json.dumps(payload)}\n\n"
                            payload = {'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
                            yield f"data: {json.dumps(payload)}\n\n"; yield "data: [DONE]\n\n"; return
                        try:
                            dat = json.loads(payload_str).get("data", {})
                        except (json.JSONDecodeError, AttributeError): continue
                        
                        delta = dat.get("delta_content", "")
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
                            if delta and settings.SHOW_THINK_TAGS:
                                if not think_open:
                                    open_payload = {'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '<think>'}, 'finish_reason': None}]}
                                    yield f"data: {json.dumps(open_payload)}\n\n"
                                    think_open = True
                                text_to_yield = self._clean_thinking(delta)
                        elif current_content_phase == "answer":
                            text_to_yield = self._clean_answer(delta)

                        if text_to_yield:
                            content_payload = {"id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": req.model, "choices": [{"index": 0, "delta": {"content": text_to_yield}, "finish_reason": None}],}
                            yield f"data: {json.dumps(content_payload)}\n\n"
        except httpx.RequestError as e:
            if ck:
                # FIX: Using the correct method name `mark_cookie_failed`
                await cookie_manager.mark_cookie_failed(ck)
            logger.error(f"Request error: {e}"); err_msg = f"Connection error: {e}"
            err = {"id": f"chatcmpl-{uuid.uuid4().hex[:29]}", "choices": [{"delta": {"content": err_msg}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"
        except Exception:
            logger.exception("Unexpected error in stream_proxy_response")
            err = {"id": f"chatcmpl-{uuid.uuid4().hex[:29]}", "choices": [{"delta": {"content": "Internal error in stream"}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"

    # ---------- non-stream ----------
    async def non_stream_proxy_response(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        ck = None
        try:
            body, headers, ck = await self._prep_upstream(req)
            full_content = []
            phase_cur = None

            async with self.client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    # FIX: Using the correct method name `mark_cookie_failed`
                    await cookie_manager.mark_cookie_failed(ck)
                    error_detail = await resp.text()
                    raise HTTPException(resp.status_code, f"Upstream error: {error_detail}")

                # If the request was successful, the cookie is good.
                await cookie_manager.mark_cookie_success(ck)

                # The robust logic for collecting chunks from the last attempt
                async for raw in resp.aiter_text():
                    for line in raw.strip().split('\n'):
                        line = line.strip()
                        if not line or not line.startswith('data: '): continue
                        payload_str = line[6:]
                        if payload_str == '[DONE]': break
                        try:
                            dat = json.loads(payload_str).get("data", {})
                        except (json.JSONDecodeError, AttributeError): continue
                        
                        delta = dat.get("delta_content")
                        new_phase = dat.get("phase")
                        
                        if new_phase:
                            phase_cur = new_phase
                        
                        if delta and phase_cur:
                            full_content.append((phase_cur, delta))
                    else: continue
                    break
            
            think_buf = []
            answer_buf = []
            for phase, content in full_content:
                if phase == "thinking":
                    think_buf.append(self._clean_thinking(content))
                elif phase == "answer":
                    answer_buf.append(self._clean_answer(content))
            
            ans_text = ''.join(answer_buf)
            final_content = ans_text

            if settings.SHOW_THINK_TAGS and think_buf:
                think_text = ''.join(think_buf).strip()
                if think_text:
                    newline = "\n" if ans_text and not ans_text.startswith(('\n', '\r')) else ""
                    final_content = f"<think>{think_text}</think>{newline}{ans_text}"

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}", created=int(time.time()), model=req.model,
                choices=[{"index": 0, "message": {"role": "assistant", "content": final_content}, "finish_reason": "stop"}],
            )
        except httpx.RequestError as e:
            if ck:
                # FIX: Using the correct method name `mark_cookie_failed`
                await cookie_manager.mark_cookie_failed(ck)
            logger.error(f"Non-stream request error: {e}"); raise HTTPException(502, f"Connection error: {e}")
        except Exception:
            logger.exception("Non-stream unexpected error"); raise HTTPException(500, "Internal server error")

    # FastAPI entry point remains the same
    async def handle_chat_completion(self, req: ChatCompletionRequest):
        stream = bool(req.stream) if req.stream is not None else settings.DEFAULT_STREAM
        if stream:
            return StreamingResponse(self.stream_proxy_response(req), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
        return await self.non_stream_proxy_response(req)