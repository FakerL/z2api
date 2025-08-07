"""
Z.AI Proxy - OpenAI-compatible API for Z.AI
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings
from models import ChatCompletionRequest, ModelsResponse, ModelInfo
from proxy_handler import ProxyHandler  # 確保 proxy_handler.py 在同級目錄
from cookie_manager import cookie_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- MODIFICATION START ---
# 1. 創建一個全局變量來持有 handler 實例
proxy_handler: ProxyHandler | None = None
# --- MODIFICATION END ---

# Security
security = HTTPBearer(auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global proxy_handler
    logger.info("Application startup: Initializing ProxyHandler and starting background tasks...")

    # --- MODIFICATION START ---
    # 2. 在應用啟動時初始化 ProxyHandler
    proxy_handler = ProxyHandler()
    # --- MODIFICATION END ---

    # Start background tasks
    health_check_task = asyncio.create_task(cookie_manager.periodic_health_check())
    
    try:
        yield
    finally:
        # Cleanup
        logger.info("Application shutdown: Cleaning up resources...")
        health_check_task.cancel()
        try:
            await health_check_task
        except asyncio.CancelledError:
            logger.info("Cookie health check task cancelled.")
        
        # --- MODIFICATION START ---
        # 3. 在應用關閉時，安全地關閉 ProxyHandler 的 client
        if proxy_handler:
            await proxy_handler.aclose()
            logger.info("ProxyHandler client closed.")
        # --- MODIFICATION END ---

# Create FastAPI app
app = FastAPI(
    title="Z.AI Proxy",
    description="OpenAI-compatible API proxy for Z.AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def verify_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication with fixed API key"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")

    if credentials.credentials != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return credentials.credentials

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    models = [
        ModelInfo(
            id=model,
            object="model",
            owned_by="z-ai"
        ) for model in settings.UPSTREAM_MODELS.keys()
    ]
    return ModelsResponse(data=models)

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _auth_token: str = Depends(verify_auth)  # 變數名加底線表示未使用
):
    """Create chat completion"""
    try:
        if not settings or not settings.COOKIES:
            raise HTTPException(
                status_code=503,
                detail="Service unavailable: No Z.AI cookies configured. Please set Z_AI_COOKIES environment variable."
            )

        if request.model not in settings.UPSTREAM_MODELS:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model}' not found."
            )

        # --- MODIFICATION START ---
        # 4. 直接使用全局的 proxy_handler 實例，並移除 async with 塊
        if not proxy_handler:
            # 這種情況理論上不應發生，因為 lifespan 會先執行
            logger.error("Proxy handler is not initialized.")
            raise HTTPException(status_code=503, detail="Service is not ready.")

        return await proxy_handler.handle_chat_completion(request)
        # --- MODIFICATION END ---

    except HTTPException:
        raise
    except Exception as e:
        # 使用 logger.exception 可以記錄完整的 traceback
        logger.exception(f"Unexpected error in chat_completions endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "invalid_request_error",
                "code": exc.status_code
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )
