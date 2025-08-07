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

    async def __aenter__(self): return self
    async def __aexit__(self, *_): await self.client.aclose()

    # ---------- text utilities ----------
    def _balance_think_tag(self, txt: str) -> str:
        # ensure opens and closes count match
        opens = len(re.findall(r"<think>", txt))
        closes = len(re.findall(r"</think>", txt))
        if opens > closes:
            txt += "</think>" * (opens - closes)
        elif closes > opens:
            for _ in range(closes - opens):
                txt = re.sub(r"</think>(?!.*</think>)", "", txt, 1)
        return txt

    def _clean_thinking(self, s: str) -> str:
        # strip html but keep text, be more careful about content preservation
        if not s: return s
        # Remove details/summary blocks
        s = re.sub(r'<details[^>]*>.*?</details>', '', s, flags=re.DOTALL)
        s = re.sub(r'<summary[^>]*>.*?</summary>', '', s, flags=re.DOTALL)
        # Remove other HTML tags but preserve content
        s = re.sub(r'<[^>]+>', '', s)
        # Clean up markdown-style quotes at line start
        s = re.sub(r'^\s*>\s*', '', s, flags=re.MULTILINE)
        return s.strip()

    def _clean_answer(self, s: str) -> str:
        # remove <details> blocks but preserve other content
        if not s: return s
        return re.sub(r"<details[^>]*>.*?</details>", "", s, flags=re.DOTALL)

    def _serialize_msgs(self, msgs) -> list:
        # convert messages to dict
        out = []
        for m in msgs:
            if hasattr(m, "dict"): out.append(m.dict())
            elif hasattr(m, "model_dump"): out.append(m.model_dump())
            elif isinstance(m, dict): out.append(m)
            else: out.append({"role": getattr(m, "role", "user"), "content": getattr(m, "content", str(m))})
        return out

    # ---------- upstream ----------
    async def _prep_upstream(self, req: ChatCompletionRequest) -> Tuple[Dict[str, Any], Dict[str, str], str]:
        ck = await cookie_manager.get_next_cookie()
        if not ck: raise HTTPException(503, "No available cookies")

        model = settings.UPSTREAM_MODEL if req.model == settings.MODEL_NAME else req.model
        body = {
            "stream": True,
            "model": model,
            "messages": self._serialize_msgs(req.messages),
            "background_tasks": {"title_generation": True, "tags_generation": True},
            "chat_id": str(uuid.uuid4()),
            "features": {
                "image_generation": False, "code_interpreter": False,
                "web_search": False, "auto_web_search": False, "enable_thinking": True,
            },
            "id": str(uuid.uuid4()),
            "mcp_servers": ["deep-web-search"],
            "model_item": {"id": model, "name": "GLM-4.5", "owned_by": "openai"},
            "params": {},
            "tool_servers": [],
            "variables": {
                "{{USER_NAME}}": "User",
                "{{USER_LOCATION}}": "Unknown",
                "{{CURRENT_DATETIME}}": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ck}",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/event-stream",
            "Accept-Language": "zh-CN",
            "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "x-fe-version": "prod-fe-1.0.53",
            "Origin": "https://chat.z.ai",
            "Referer": "https://chat.z.ai/",
        }
        return body, headers, ck

    # ---------- stream ----------
    async def stream_proxy_response(self, req: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        # 使用 self.client 而不是每次都創建新的 client，以利用連接池
        try:
            body, headers, ck = await self._prep_upstream(req)
            comp_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            think_open, phase_cur = False, None
            # FIX: 移除了 first_answer_chunk 標誌，改用更可靠的狀態管理

            async with self.client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers, timeout=self.client.timeout) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(ck)
                    err_msg = f"Upstream error: {resp.status_code}"
                    try:
                        err_body = await resp.aread()
                        err_msg += f" - {err_body.decode()}"
                    except Exception:
                        pass
                    err = {
                        "id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {"content": err_msg}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"; return

                async for raw in resp.aiter_text():
                    if not raw or raw.isspace(): continue
                    for line in raw.split('\n'):
                        line = line.strip()
                        if not line or line.startswith(':') or not line.startswith('data: '): continue

                        payload = line[6:]
                        if payload == '[DONE]':
                            if think_open: # 如果 <think> 標籤仍然是打開的，在結束前關閉它
                                close_c = {
                                    "id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()),
                                    "model": req.model,
                                    "choices": [{"index": 0, "delta": {"content": "</think>"}, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(close_c)}\n\n"
                                think_open = False
                            
                            final_c = {
                                "id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()),
                                "model": req.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            }
                            yield f"data: {json.dumps(final_c)}\n\n"; yield "data: [DONE]\n\n"; return

                        try:
                            parsed = json.loads(payload)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode JSON payload: {payload}")
                            continue
                        
                        dat = parsed.get("data", {})
                        delta, phase = dat.get("delta_content", ""), dat.get("phase")

                        # --- 主要修復邏輯開始 ---
                        # 1. 處理階段變化 (Phase Transition)
                        if phase and phase != phase_cur:
                            # 當從 'thinking' 切換到 'answer' 時，關閉 <think> 標籤
                            if phase_cur == "thinking" and phase == "answer":
                                if think_open and settings.SHOW_THINK_TAGS:
                                    close_c = {
                                        "id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()),
                                        "model": req.model,
                                        "choices": [{"index": 0, "delta": {"content": "</think>\n"}, "finish_reason": None}],
                                    }
                                    yield f"data: {json.dumps(close_c)}\n\n"
                                    think_open = False
                            phase_cur = phase

                        # 2. 處理內容 (Content Processing)
                        # 這個邏輯塊獨立於階段變化，確保當前 chunk 的內容總是被處理
                        text_to_yield = ""
                        if phase_cur == "thinking":
                            if settings.SHOW_THINK_TAGS:
                                # 如果 think 標籤還沒打開，就打開它
                                if not think_open:
                                    open_c = {
                                        "id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()),
                                        "model": req.model,
                                        "choices": [{"index": 0, "delta": {"content": "<think>"}, "finish_reason": None}],
                                    }
                                    yield f"data: {json.dumps(open_c)}\n\n"
                                    think_open = True
                                text_to_yield = self._clean_thinking(delta)

                        elif phase_cur == "answer":
                            text_to_yield = self._clean_answer(delta)

                        # 3. 發送內容 (Yield Content)
                        # 只有在 text_to_yield 有實際內容時才發送，避免發送空 chunk
                        if text_to_yield:
                            out = {
                                "id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()),
                                "model": req.model,
                                "choices": [{"index": 0, "delta": {"content": text_to_yield}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(out)}\n\n"
                        # --- 主要修復邏輯結束 ---

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            err = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:29]}", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": req.model,
                "choices": [{"index": 0, "delta": {"content": f"Connection error: {e}"}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception(f"Unexpected error in stream_proxy_response: {e}") # 使用 exception 打印 traceback
            err = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:29]}", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": req.model,
                "choices": [{"index": 0, "delta": {"content": f"Internal server error."}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"

    # ---------- non-stream ----------
    async def non_stream_proxy_response(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        # 非流式邏輯本身比較穩健，因為它先收集所有數據再處理。
        # 此處的邏輯已是最佳實踐，無需大改。
        ck = None # 在 try 外部定義，以便 finally 中可以訪問
        try:
            body, headers, ck = await self._prep_upstream(req)
            think_buf, answer_buf = [], []
            
            # 確保使用實例的 client 和其 timeout 設置
            async with self.client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers, timeout=self.client.timeout) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(ck)
                    error_detail = await resp.text()
                    logger.error(f"Upstream error {resp.status_code}: {error_detail}")
                    raise HTTPException(resp.status_code, f"Upstream error: {error_detail}")

                async for raw in resp.aiter_text():
                    if not raw or raw.isspace(): continue
                    for line in raw.split('\n'):
                        line = line.strip()
                        if not line or line.startswith(':') or not line.startswith('data: '): continue
                        payload = line[6:]
                        if payload == '[DONE]': break
                        try: 
                            parsed = json.loads(payload)
                        except json.JSONDecodeError: 
                            continue
                        dat = parsed.get("data", {})
                        delta, phase = dat.get("delta_content", ""), dat.get("phase")
                        
                        if not delta: continue
                        
                        if phase == "thinking": 
                            think_buf.append(delta)
                        elif phase == "answer": 
                            answer_buf.append(delta)
                # 循環結束後 break
                else: # for-else 語句，如果 for 循環正常結束（非 break），則執行
                    pass # 此處不需要做任何事

            # 合併內容後再進行清理
            raw_answer = ''.join(answer_buf)
            ans_text = self._clean_answer(raw_answer)

            final_content = ans_text
            if settings.SHOW_THINK_TAGS and think_buf:
                raw_thinking = ''.join(think_buf)
                think_text = self._clean_thinking(raw_thinking)
                # 確保 thinking 內容不為空時才添加標籤和換行
                if think_text:
                    final_content = f"<think>{think_text}</think>\n{ans_text}"

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
                created=int(time.time()),
                model=req.model,
                choices=[{"index": 0, "message": {"role": "assistant", "content": final_content}, "finish_reason": "stop"}],
                # 可以在此添加 usage 信息，如果 API 返回的話
            )
        except httpx.RequestError as e:
            logger.error(f"Non-stream request error: {e}")
            if ck: await cookie_manager.mark_cookie_invalid(ck)
            raise HTTPException(502, f"Connection error to upstream: {e}")
        except Exception as e:
            logger.exception(f"Non-stream unexpected error: {e}") # 使用 exception 打印 traceback
            raise HTTPException(500, "Internal server error")


    # ---------- FastAPI entry ----------
    async def handle_chat_completion(self, req: ChatCompletionRequest):
        # 移除對 self.client 的重複創建，改用 __aenter__ 和 __aexit__
        # 在 FastAPI 中，通常使用 Depends 來管理依賴項的生命週期
        # 但這裡 ProxyHandler 作為一個普通類，這樣的寫法也是可以的
        stream = bool(req.stream) if req.stream is not None else settings.DEFAULT_STREAM
        if stream:
            return StreamingResponse(
                self.stream_proxy_response(req),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return await self.non_stream_proxy_response(req)