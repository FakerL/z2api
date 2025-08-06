"""
Proxy handler for Z.AI API requests (OpenAI-compatible)
"""
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, Any, Optional

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from cookie_manager import cookie_manager
from models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)

logger = logging.getLogger(__name__)


class ProxyHandler:
    def __init__(self):
        # Z.AI 端連線逾時 60 秒
        self.client = httpx.AsyncClient(timeout=60.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    # --------- 文字前處理 ---------
    def transform_content(self, content: str) -> str:
        """
        依照專案設定將 Z.AI 傳回的 HTML / THINK TAG 等轉成純文字
        """
        if not content:
            return content
        # 例：過濾 <br/> 與 <think>
        content = content.replace("<br/>", "\n")
        if not settings.SHOW_THINK_TAGS:
            content = content.replace("<think>", "").replace("</think>", "")
        return content.strip()

    # --------- 主要進入點 ---------
    async def proxy_request(self, request: ChatCompletionRequest):
        """
        OpenAI API 相容的 proxy 入口
        """
        cookie = await cookie_manager.get_next_cookie()
        if not cookie:
            raise HTTPException(status_code=503, detail="No available cookies")

        # 若對外聲稱的 model 與內部實際 model 不同，在此轉換
        target_model = (
            settings.UPSTREAM_MODEL
            if request.model == settings.MODEL_NAME
            else request.model
        )

        # 決定是否串流
        is_streaming: bool = (
            request.stream if request.stream is not None else settings.DEFAULT_STREAM
        )

        # 向 Z.AI 串流或一次性取資料
        if is_streaming:
            # 建立 SSE StreamingResponse
            return StreamingResponse(
                self.stream_response(request, target_model, cookie),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    # 若有 nginx 建議加上 X-Accel-Buffering: no
                },
            )
        else:
            # 非串流：拿到完整內容後包成 ChatCompletionResponse
            content = await self.get_full_response(request, target_model, cookie)
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=target_model,
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            )

    # --------- 非串流邏輯 ---------
    async def get_full_response(
        self,
        request: ChatCompletionRequest,
        target_model: str,
        cookie: str,
    ) -> str:
        """
        向 Z.AI 取完整回覆並回傳轉換後文字
        """
        resp = await self.client.post(
            settings.ZAI_ENDPOINT,
            headers={"Cookie": cookie},
            json=request.model_dump(exclude_none=True),
        )
        resp.raise_for_status()
        data = resp.json()
        return self.transform_content(data["choices"][0]["message"]["content"])

    # --------- 串流邏輯 ---------
    async def stream_response(
        self,
        request: ChatCompletionRequest,
        target_model: str,
        cookie: str,
    ) -> AsyncGenerator[str, None]:
        """
        將 Z.AI 串流資料即時轉成 OpenAI SSE 片段
        """
        # 呼叫 Z.AI 串流端點（假設支援 HTTP chunk）
        async with self.client.stream(
            "POST",
            settings.ZAI_STREAM_ENDPOINT,
            headers={"Cookie": cookie},
            json=request.model_dump(exclude_none=True),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                # Z.AI 每行可能已是 json；自行視格式解析
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("skip non-json line from Z.AI: %s", line)
                    continue

                # 取得文字增量
                delta_text = self.transform_content(raw.get("delta", ""))
                if delta_text == "":
                    continue

                # 組成 OpenAI stream chunk
                chunk: Dict[str, Any] = {
                    "id": raw.get("id", f"chatcmpl-{uuid.uuid4()}"),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": target_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta_text},
                            "finish_reason": None,
                        }
                    ],
                }

                # 送出 SSE formatted line
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # Z.AI 結束後送出 [DONE]
            yield "data: [DONE]\n\n"