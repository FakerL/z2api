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
    async def __aexit__(self, exc_type, exc_val, exc_tb): await self.client.aclose()

    # ---------- text utilities ----------
    def _balance_think_tag(self, txt: str) -> str:
        opens, closes = len(re.findall(r"

<details type="reasoning" done="true" duration="0">
<summary>Thought for 0 seconds</summary>
> ", txt)), len(re.findall(r"
</details>
", txt))
        if opens > closes: txt += "</think>" * (opens - closes)
        elif closes > opens:
            for _ in range(closes - opens):
                txt = re.sub(r"</think>(?!</think>)(?![^<]*</think>)$", "", txt, 1)
        return txt

    def _clean_thinking(self, s: str) -> str:
        if not s: return s
        s = re.sub(r'<details[^>]*>|</details>|<summary[^>]*>.*?</summary>', '', s, flags=re.DOTALL)
        s = re.sub(r'<[^>]+>', '', s)
        s = re.sub(r'^>\s*|\n>\s*', '\n', s)
        return s.strip()

    def _clean_answer(self, s: str) -> str:
        return re.sub(r"<details[^>]*>.*?</details>", "", s, flags=re.DOTALL) if s else s

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
                "image_generation": False, "code_interpreter": False, "web_search": False,
                "auto_web_search": False, "enable_thinking": True,
            },
            "id": str(uuid.uuid4()),
            "mcp_servers": ["deep-web-search"],
            "model_item": {"id": model, "name": "GLM-4.5", "owned_by": "openai"},
            "params": {}, "tool_servers": [],
            "variables": {
                "{{USER_NAME}}": "User", "{{USER_LOCATION}}": "Unknown",
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
        client = None
        try:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, read=300.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                http2=True,
            )
            body, headers, ck = await self._prep_upstream(req)
            comp_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            think_open, phase_cur, need_nl = False, None, False

            async with client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(ck)
                    err = {
                        "id": comp_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {"content": f"Error: {resp.status_code}"}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"; return

                async for raw in resp.aiter_text():
                    if not raw or raw.isspace(): continue
                    for line in raw.split('\n'):
                        line = line.strip()
                        if not line or line.startswith(':') or not line.startswith('data: '): continue
                        payload = line[6:]
                        if payload == '[DONE]':
                            if think_open and settings.SHOW_THINK_TAGS:
                                close_c = {
                                    "id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()),
                                    "model": req.model,
                                    "choices": [{"index": 0, "delta": {"content": "</think>"}, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(close_c)}\n\n"
                            final_c = {
                                "id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()),
                                "model": req.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            }
                            yield f"data: {json.dumps(final_c)}\n\n"; yield "data: [DONE]\n\n"; return

                        try:
                            parsed = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        dat = parsed.get("data", {})
                        delta, phase = dat.get("delta_content", ""), dat.get("phase")

                        if phase != phase_cur:
                            phase_cur = phase
                            if phase == "answer" and think_open and settings.SHOW_THINK_TAGS:
                                auto_close = {
                                    "id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()),
                                    "model": req.model,
                                    "choices": [{"index": 0, "delta": {"content": "</think>"}, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(auto_close)}\n\n"
                                think_open, need_nl = False, True  # add \n before first answer

                        if phase == "thinking" and not settings.SHOW_THINK_TAGS: continue
                        if not delta: continue

                        if phase == "thinking" and settings.SHOW_THINK_TAGS:
                            if not think_open:
                                tk = {
                                    "id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()),
                                    "model": req.model,
                                    "choices": [{"index": 0, "delta": {"content": "<think>"}, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(tk)}\n\n"; think_open = True
                            text = self._clean_thinking(delta)
                        else:
                            if need_nl:
                                delta = "\n" + delta
                                need_nl = False
                            text = self._clean_answer(delta)

                        if text:
                            out = {
                                "id": comp_id, "object": "chat.completion.chunk", "created": int(time.time()),
                                "model": req.model,
                                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(out)}\n\n"

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            err = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": f"Connection error: {e}"}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            err = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": f"Internal error: {e}"}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(err)}\n\n"; yield "data: [DONE]\n\n"
        finally:
            if client: await client.aclose()

    # ---------- non-stream ----------
    async def non_stream_proxy_response(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        client = None
        try:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, read=300.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                http2=True,
            )
            body, headers, ck = await self._prep_upstream(req)
            think_buf, answer_buf, phase_cur = [], [], None

            async with client.stream("POST", settings.UPSTREAM_URL, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    await cookie_manager.mark_cookie_invalid(ck)
                    raise HTTPException(resp.status_code, "Upstream error")
                async for raw in resp.aiter_text():
                    if not raw or raw.isspace(): continue
                    for line in raw.split('\n'):
                        line = line.strip()
                        if not line or line.startswith(':') or not line.startswith('data: '): continue
                        payload = line[6:]
                        if payload == '[DONE]': break
                        try: parsed = json.loads(payload)
                        except json.JSONDecodeError: continue
                        dat = parsed.get("data", {})
                        delta, phase = dat.get("delta_content", ""), dat.get("phase")
                        if not delta: continue
                        if phase == "thinking": think_buf.append(delta)
                        elif phase == "answer": answer_buf.append(delta)

            ans_text = self._clean_answer(''.join(answer_buf))
            if settings.SHOW_THINK_TAGS and think_buf:
                think_text = self._clean_thinking(''.join(think_buf))
                final = f"<think>{think_text}</think>\n{ans_text}"
            else:
                final = ans_text

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
                created=int(time.time()),
                model=req.model,
                choices=[{"index": 0, "message": {"role": "assistant", "content": final}, "finish_reason": "stop"}],
            )
        except httpx.RequestError as e:
            logger.error(f"Non-stream request error: {e}")
            if "ck" in locals(): await cookie_manager.mark_cookie_invalid(ck)
            raise HTTPException(502, "Connection error")
        except Exception as e:
            logger.error(f"Non-stream unexpected error: {e}")
            raise HTTPException(500, "Internal error")
        finally:
            if client: await client.aclose()

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