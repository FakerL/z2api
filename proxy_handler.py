"""
Proxy handler for Z.AI API requests
"""
import json, logging, re, time, uuid
from typing import AsyncGenerator, Dict, Any, Tuple

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

# 確保這些導入與您的項目結構匹配
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

    # _balance_think_tag, _clean_thinking, _clean_answer, _serialize_msgs, _prep_upstream 保持不變
    def _balance_think_tag(self, txt: str) -> str:
        opens = len(re.findall(r"<think>", txt))
        closes = len(re.findall(r"</think>", txt))
        if opens > closes: txt += "</think>" * (opens - closes)
        elif closes > opens:
            for _ in range(closes - opens): txt = re.sub(r"</think>(?!.*</think>)", "", txt, 1)
        return txt
    def _clean_thinking(self, s: str) -> str:
        if not s: return s
        s = re.sub(r'<details[^>]*>.*?</details>', '', s, flags=re.DOTALL)
        s = re.sub(r'<summary[^>]*>.*?</summary>', '', s, flags=re.DOTALL)
        s = re.sub(r'<[^>]+>', '', s)
        s = re.sub(r'^\s*>\s*', '', s, flags=re.MULTILINE)
        return s.strip()
    def _clean_answer(self, s: str) -> str:
        if not s: return s
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
                    await cookie_manager.mark_cookie_invalid(ck)
                    # ... (error handling code remains the same)
                    err_body = await resp.aread()
                    err_msg = f"Error: {resp.status_code} - {err_body.decode(errors='ignore')}"
                    err = {"id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": req.model, "choices": [{"index": 0, "delta": {"content": err_msg}, "finish_reason": "stop"}],}
                    yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"; return

                async for raw in resp.aiter_text():
                    for line in raw.strip().split('\n'):
                        line = line.strip()
                        if not line or not line.startswith('data: '): continue

                        payload_str = line[6:]
                        if payload_str == '[DONE]':
                            if think_open:
                                close_payload = {'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '</think>'}, 'finish_reason': None}]}
                                yield f"data: {json.dumps(close_payload)}\n\n"
                            final_payload = {'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
                            yield f"data: {json.dumps(final_payload)}\n\n"
                            yield "data: [DONE]\n\n"; return

                        try:
                            parsed = json.loads(payload_str)
                            dat = parsed.get("data", {})
                        except json.JSONDecodeError:
                            continue

                        # --- FIX START: REVISED LOGIC ---
                        delta = dat.get("delta_content", "")
                        new_phase = dat.get("phase")

                        # Step 1: Determine the effective phase for THIS chunk
                        effective_phase = new_phase or phase_cur

                        # Step 2: Handle phase transition side-effects (closing tags) BEFORE processing content
                        if new_phase and new_phase != phase_cur:
                            if phase_cur == "thinking" and new_phase == "answer":
                                if think_open and settings.SHOW_THINK_TAGS:
                                    close_payload = {'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '</think>\n'}, 'finish_reason': None}]}
                                    yield f"data: {json.dumps(close_payload)}\n\n"
                                    think_open = False
                            # Update phase_cur AFTER handling the transition
                            phase_cur = new_phase
                        
                        # Step 3: Process the delta content of THIS chunk based on its effective phase
                        text_to_yield = ""
                        if effective_phase == "thinking":
                            if settings.SHOW_THINK_TAGS:
                                if not think_open:
                                    open_payload = {'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '<think>'}, 'finish_reason': None}]}
                                    yield f"data: {json.dumps(open_payload)}\n\n"
                                    think_open = True
                                text_to_yield = self._clean_thinking(delta)

                        elif effective_phase == "answer":
                            text_to_yield = self._clean_answer(delta)
                        
                        # Step 4: Yield the processed content if it exists
                        if text_to_yield:
                            chunk_payload = {"id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": req.model, "choices": [{"index": 0, "delta": {"content": text_to_yield}, "finish_reason": None}],}
                            yield f"data: {json.dumps(chunk_payload)}\n\n"
                        # --- FIX END ---
        except httpx.RequestError as e:
            # ... (error handling code remains the same)
            if ck: await cookie_manager.mark_cookie_invalid(ck)
            logger.error(f"Request error: {e}")
            err_msg = f"Connection error: {e}"
            err = {"id": f"chatcmpl-{uuid.uuid4().hex[:29]}", "choices": [{"delta": {"content": err_msg}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception(f"Unexpected error in stream_proxy_response")
            err = {"id": f"chatcmpl-{uuid.uuid4().hex[:29]}", "choices": [{"delta": {"content": f"Internal error in stream"}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"


    # ---------- non-stream ----------
    async def non_stream_proxy_response(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        ck = None
        try:
            body, headers, ck = await self._prep_upstream(req)
            think_buf, answer_buf = [], []
            phase_cur = None

            async with self.client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(ck)
                    error_detail = await resp.text()
                    raise HTTPException(resp.status_code, f"Upstream error: {error_detail}")

                async for raw in resp.aiter_text():
                    for line in raw.strip().split('\n'):
                        line = line.strip()
                        if not line or not line.startswith('data: '): continue
                        
                        payload_str = line[6:]
                        if payload_str == '[DONE]': break
                        
                        try:
                            parsed = json.loads(payload_str)
                            dat = parsed.get("data", {})
                        except json.JSONDecodeError:
                            continue
                        
                        # --- FIX START: REVISED LOGIC ---
                        delta = dat.get("delta_content", "")
                        new_phase = dat.get("phase")

                        # Step 1: Update the current phase if a new one is provided
                        if new_phase:
                            phase_cur = new_phase
                        
                        # Step 2: Append the delta to the correct buffer based on the updated phase
                        if not delta: continue
                        
                        if phase_cur == "thinking":
                            think_buf.append(delta)
                        elif phase_cur == "answer":
                            answer_buf.append(delta)
                        # --- FIX END ---
                    else: # continue for inner loop
                        continue
                    break # break for outer loop
            
            raw_answer = ''.join(answer_buf)
            ans_text = self._clean_answer(raw_answer)

            final_content = ans_text
            if settings.SHOW_THINK_TAGS and think_buf:
                raw_thinking = ''.join(think_buf)
                think_text = self._clean_thinking(raw_thinking)
                if think_text:
                    final_content = f"<think>{think_text}</think>\n{ans_text}"

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}", created=int(time.time()), model=req.model,
                choices=[{"index": 0, "message": {"role": "assistant", "content": final_content}, "finish_reason": "stop"}],
            )
        except httpx.RequestError as e:
            # ... (error handling code remains the same)
            if ck: await cookie_manager.mark_cookie_invalid(ck)
            logger.error(f"Non-stream request error: {e}")
            raise HTTPException(502, f"Connection error: {e}")
        except Exception as e:
            logger.exception(f"Non-stream unexpected error")
            raise HTTPException(500, "Internal server error")


    # ---------- FastAPI entry ----------
    async def handle_chat_completion(self, req: ChatCompletionRequest):
        stream = bool(req.stream) if req.stream is not None else settings.DEFAULT_STREAM
        if stream:
            return StreamingResponse(
                self.stream_proxy_response(req),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return await self.non_stream_proxy_response(req)