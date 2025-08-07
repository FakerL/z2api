"""
Proxy handler for Z.AI API requests - DEBUG VERSION
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
# Set logger to DEBUG level to ensure all our messages are captured
logger.setLevel(logging.DEBUG)


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

    # --- Other methods ---
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

    # Streaming response is left as-is for now, we focus on non-stream for debugging
    async def stream_proxy_response(self, req: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        # This function remains the same as the previous correct version.
        # The focus of debugging is on the non-stream version.
        ck = None
        try:
            body, headers, ck = await self._prep_upstream(req)
            comp_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"; think_open = False; phase_cur = None
            async with self.client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_failed(ck)
                    err_body = await resp.aread(); err_msg = f"Error: {resp.status_code} - {err_body.decode(errors='ignore')}"
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
                        try: dat = json.loads(payload_str).get("data", {})
                        except (json.JSONDecodeError, AttributeError): continue
                        delta = dat.get("delta_content", ""); new_phase = dat.get("phase")
                        is_transition = new_phase and new_phase != phase_cur
                        if is_transition:
                            if phase_cur == "thinking" and new_phase == "answer" and think_open and settings.SHOW_THINK_TAGS:
                                yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '</think>'}, 'finish_reason': None}]})}\n\n"
                                think_open = False
                            phase_cur = new_phase
                        current_content_phase = phase_cur or new_phase
                        text_to_yield = ""
                        if current_content_phase == "thinking":
                            if delta and settings.SHOW_THINK_TAGS:
                                if not think_open:
                                    yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '<think>'}, 'finish_reason': None}]})}\n\n"
                                    think_open = True
                                text_to_yield = self._clean_thinking(delta)
                        elif current_content_phase == "answer": text_to_yield = self._clean_answer(delta)
                        if text_to_yield: yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': text_to_yield}, 'finish_reason': None}]})}\n\n"
        except Exception: logger.exception("Stream error"); raise

    # ---------- NON-STREAM (DEBUG VERSION) ----------
    async def non_stream_proxy_response(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        logger.debug("="*20 + " STARTING NON-STREAM DEBUG " + "="*20)
        ck = None
        try:
            body, headers, ck = await self._prep_upstream(req)
            full_content_stream = []
            phase_cur = None

            async with self.client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_failed(ck)
                    error_detail = await resp.text()
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
                            full_content_stream.append(dat) # Store the raw parsed data
                        except (json.JSONDecodeError, AttributeError): continue
                    else: continue
                    break

            # --- STAGE 1: LOG RAW COLLECTED DATA ---
            logger.debug("-" * 10 + " STAGE 1: RAW DATA FROM Z.AI " + "-" * 10)
            for i, dat in enumerate(full_content_stream):
                # Use !r to get an unambiguous representation of the string
                logger.debug(f"Chunk {i}: {dat!r}")
            logger.debug("-" * 10 + " END STAGE 1 " + "-" * 10)


            # --- STAGE 2: PROCESS AND LOG RAW JOINED STRINGS ---
            raw_think_str = ''.join([d.get("delta_content", "") for d in full_content_stream if d.get("phase") == "thinking"])
            raw_answer_str = ''.join([d.get("delta_content", "") for d in full_content_stream if d.get("phase") == "answer"])
            
            # This is a fallback for chunks that might not have a phase.
            # It's more complex but might catch the edge case.
            phase_aware_think = []
            phase_aware_answer = []
            current_phase_for_build = None
            for d in full_content_stream:
                if 'phase' in d:
                    current_phase_for_build = d['phase']
                if 'delta_content' in d:
                    if current_phase_for_build == 'thinking':
                        phase_aware_think.append(d['delta_content'])
                    elif current_phase_for_build == 'answer':
                        phase_aware_answer.append(d['delta_content'])
            
            phase_aware_raw_answer_str = ''.join(phase_aware_answer)


            logger.debug("-" * 10 + " STAGE 2: RAW JOINED STRINGS " + "-" * 10)
            logger.debug(f"Phase-unaware Think String: {raw_think_str!r}")
            logger.debug(f"Phase-unaware Answer String: {raw_answer_str!r}")
            logger.debug(f"Phase-aware Answer String: {phase_aware_raw_answer_str!r}")
            logger.debug("-" * 10 + " END STAGE 2 " + "-" * 10)


            # --- STAGE 3: PROCESS AND LOG FINAL CLEANED TEXT ---
            # We will use the more robust phase-aware string for the final result
            think_text = self._clean_thinking(''.join(phase_aware_think)).strip()
            ans_text = self._clean_answer(''.join(phase_aware_answer))


            logger.debug("-" * 10 + " STAGE 3: FINAL CLEANED TEXT " + "-" * 10)
            logger.debug(f"Final Think Text: {think_text!r}")
            logger.debug(f"Final Answer Text: {ans_text!r}")
            logger.debug("-" * 10 + " END STAGE 3 " + "-" * 10)


            # Final construction
            final_content = ans_text
            if settings.SHOW_THINK_TAGS and think_text:
                newline = "\n" if ans_text and not ans_text.startswith(('\n', '\r')) else ""
                final_content = f"<think>{think_text}</think>{newline}{ans_text}"
            
            logger.debug(f"Final constructed content to be returned: {final_content!r}")
            logger.debug("="*20 + " END NON-STREAM DEBUG " + "="*20)


            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}", created=int(time.time()), model=req.model,
                choices=[{"index": 0, "message": {"role": "assistant", "content": final_content}, "finish_reason": "stop"}],
            )
        except Exception:
            logger.exception("Non-stream processing failed")
            raise

    # ---------- FastAPI entry ----------
    async def handle_chat_completion(self, req: ChatCompletionRequest):
        # Force non-stream mode for this debug session
        if req.stream:
             logger.warning("Request specified streaming, but DEBUG handler is forcing non-stream mode.")
        
        # Make sure stream is False to trigger the debug path
        req.stream = False
        return await self.non_stream_proxy_response(req)