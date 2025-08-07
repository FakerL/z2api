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

    # FIX: 移除 __aenter__ 和 __aexit__，改用顯式的 aclose 方法
    # aenter/aexit 模式不適用於需要跨越請求生命週期的流式響應
    async def aclose(self):
        """Closes the underlying httpx client."""
        if not self.client.is_closed:
            await self.client.aclose()

    # ---------- text utilities ----------
    # _balance_think_tag, _clean_thinking, _clean_answer, _serialize_msgs 方法保持不變
    def _balance_think_tag(self, txt: str) -> str:
        opens = len(re.findall(r"<think>", txt))
        closes = len(re.findall(r"</think>", txt))
        if opens > closes:
            txt += "</think>" * (opens - closes)
        elif closes > opens:
            for _ in range(closes - opens):
                txt = re.sub(r"</think>(?!.*</think>)", "", txt, 1)
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

    # ---------- upstream ----------
    async def _prep_upstream(self, req: ChatCompletionRequest) -> Tuple[Dict[str, Any], Dict[str, str], str]:
        # 此方法保持不變
        ck = await cookie_manager.get_next_cookie()
        if not ck: raise HTTPException(503, "No available cookies")

        model = settings.UPSTREAM_MODEL if req.model == settings.MODEL_NAME else req.model
        body = {
            "stream": True, "model": model, "messages": self._serialize_msgs(req.messages),
            "background_tasks": {"title_generation": True, "tags_generation": True}, "chat_id": str(uuid.uuid4()),
            "features": {"image_generation": False, "code_interpreter": False, "web_search": False, "auto_web_search": False, "enable_thinking": True,},
            "id": str(uuid.uuid4()), "mcp_servers": ["deep-web-search"],
            "model_item": {"id": model, "name": "GLM-4.5", "owned_by": "openai"}, "params": {}, "tool_servers": [],
            "variables": {"{{USER_NAME}}": "User", "{{USER_LOCATION}}": "Unknown", "{{CURRENT_DATETIME}}": time.strftime("%Y-%m-%d %H:%M:%S"),},
        }
        headers = {
            "Content-Type": "application/json", "Authorization": f"Bearer {ck}",
            "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"),
            "Accept": "application/json, text/event-stream", "Accept-Language": "zh-CN",
            "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
            "sec-ch-ua-mobile": "?0", "sec-ch-ua-platform": '"macOS"', "x-fe-version": "prod-fe-1.0.53",
            "Origin": "https://chat.z.ai", "Referer": "https://chat.z.ai/",
        }
        return body, headers, ck

    # ---------- stream ----------
    async def stream_proxy_response(self, req: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        ck = None
        try:
            body, headers, ck = await self._prep_upstream(req)
            comp_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            think_open = False
            
            # FIX: 維護一個持久的 phase 狀態
            phase_cur = None

            async with self.client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(ck)
                    err_body = await resp.aread()
                    err_msg = f"Error: {resp.status_code} - {err_body.decode(errors='ignore')}"
                    err = {
                        "id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": req.model,
                        "choices": [{"index": 0, "delta": {"content": err_msg}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"; return

                async for raw in resp.aiter_text():
                    if not raw or raw.isspace(): continue
                    for line in raw.split('\n'):
                        line = line.strip()
                        if not line or not line.startswith('data: '): continue

                        payload = line[6:]
                        if payload == '[DONE]':
                            if think_open:
                                yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '</think>'}, 'finish_reason': None}]})}\n\n"
                            yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                            yield "data: [DONE]\n\n"; return

                        try:
                            parsed = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        
                        dat = parsed.get("data", {})
                        delta, new_phase = dat.get("delta_content", ""), dat.get("phase")

                        # FIX: 正確的狀態管理邏輯
                        # 1. 如果收到了新的 phase，更新當前 phase
                        if new_phase and new_phase != phase_cur:
                            # 處理從 thinking 到 answer 的過渡
                            if phase_cur == "thinking" and new_phase == "answer" and think_open and settings.SHOW_THINK_TAGS:
                                yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '</think>\n'}, 'finish_reason': None}]})}\n\n"
                                think_open = False
                            phase_cur = new_phase
                        
                        if not delta: continue # 如果沒有內容，則跳過

                        # 2. 根據當前的 phase_cur 處理內容
                        text_to_yield = ""
                        if phase_cur == "thinking":
                            if settings.SHOW_THINK_TAGS:
                                if not think_open:
                                    yield f"data: {json.dumps({'id': comp_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': '<think>'}, 'finish_reason': None}]})}\n\n"
                                    think_open = True
                                text_to_yield = self._clean_thinking(delta)
                        elif phase_cur == "answer":
                            text_to_yield = self._clean_answer(delta)

                        # 3. 發送處理後的內容
                        if text_to_yield:
                            out = {
                                "id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": req.model,
                                "choices": [{"index": 0, "delta": {"content": text_to_yield}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(out)}\n\n"

        except httpx.RequestError as e:
            if ck: await cookie_manager.mark_cookie_invalid(ck)
            logger.error(f"Request error: {e}")
            err_msg = f"Connection error: {e}"
            err = {"choices": [{"delta": {"content": err_msg}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception(f"Unexpected error in stream_proxy_response")
            err = {"choices": [{"delta": {"content": f"Internal error in stream"}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"

    # ---------- non-stream ----------
    async def non_stream_proxy_response(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        ck = None
        try:
            body, headers, ck = await self._prep_upstream(req)
            think_buf, answer_buf = [], []
            
            # FIX: 維護一個持久的 phase 狀態
            phase_cur = None

            async with self.client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(ck)
                    error_detail = await resp.text()
                    raise HTTPException(resp.status_code, f"Upstream error: {error_detail}")

                async for raw in resp.aiter_text():
                    if not raw or raw.isspace(): continue
                    for line in raw.split('\n'):
                        line = line.strip()
                        if not line or not line.startswith('data: '): continue
                        payload = line[6:]
                        if payload == '[DONE]': break
                        
                        try: parsed = json.loads(payload)
                        except json.JSONDecodeError: continue
                        
                        dat = parsed.get("data", {})
                        delta, new_phase = dat.get("delta_content", ""), dat.get("phase")
                        
                        # FIX: 正確的狀態管理邏輯
                        # 1. 更新當前 phase
                        if new_phase:
                            phase_cur = new_phase
                        
                        if not delta: continue
                        
                        # 2. 根據當前的 phase_cur 存儲內容
                        if phase_cur == "thinking": 
                            think_buf.append(delta)
                        elif phase_cur == "answer": 
                            answer_buf.append(delta)
                else: # for-else, will be executed if loop finishes without break
                    pass
            
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