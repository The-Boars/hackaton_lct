import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.model_wrapper import NERModel

logger = logging.getLogger("uvicorn.error")


PREDICT_TIMEOUT = float(os.getenv("PREDICT_TIMEOUT_SEC", "0.95"))
MAX_INPUT_LEN = int(os.getenv("MAX_INPUT_LEN", "512"))  # защита от слишком длинных тел

model: NERModel | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Loading NER model...")
    model = NERModel()
    logger.info("Model loaded.")
    yield
    # Ничего не освобождаем: модель реюзается до остановки процесса

app = FastAPI(title="NER Service", version="1.0.0", lifespan=lifespan)

# Разрешаем публичный доступ (для бота и внешнего URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
)

class PredictIn(BaseModel):
    # Пустая строка — вернём пустой список
    input: str = Field("", description="Текст запроса")

class EntityOut(BaseModel):
    start_index: int
    end_index: int
    entity: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/predict", response_model=List[EntityOut])
async def predict(body: PredictIn):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    text = (body.input or "").strip()
    if not text:
        return []

    # Жёстко ограничим длину входа (ускоряет худшие случаи)
    if len(text) > MAX_INPUT_LEN:
        text = text[:MAX_INPUT_LEN]

    # Вынесем синхронную инференс-функцию в пул потоков, чтобы не блокировать event loop
    def _infer_sync():
        return model.predict(text)

    try:
        # Гарантируем ответ < ~1с
        result = await asyncio.wait_for(asyncio.to_thread(_infer_sync), timeout=PREDICT_TIMEOUT)
        # Ожидается список словарей с ключами start_index, end_index, entity — пускаем как есть
        return result
    except asyncio.TimeoutError:
        # Жёстко соблюдаем SLA: не висим дольше, честно отвечаем пустым списком
        logger.warning("Predict timeout (> %.3fs) for text len=%d", PREDICT_TIMEOUT, len(text))
        return []
    except Exception as e:
        logger.exception("Predict failed: %s", e)
        # Для стабильности сервиса — 200 с пустым списком (можно поменять на 500)
        return []

@app.get("/")
async def root():
    return {"service": "NER", "docs": "/docs"}
