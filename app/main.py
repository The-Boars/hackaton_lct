import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.model_wrapper import NERModel


# Configure JSON logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(record.created)
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if they exist
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)


# Configure root logger
logging.basicConfig(level=logging.INFO)
root_logger = logging.getLogger()
root_logger.handlers.clear()

# Add console handler with JSON formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(JSONFormatter())
root_logger.addHandler(console_handler)

# Disable uvicorn access logs
logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


PREDICT_TIMEOUT = float(os.getenv("PREDICT_TIMEOUT_SEC", "0.95"))
MAX_INPUT_LEN = int(os.getenv("MAX_INPUT_LEN", "512"))  # защита от слишком длинных тел

model: NERModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(
        "Loading NER model...", extra={"extra_fields": {"event": "model_loading_start"}}
    )
    model = NERModel()
    logger.info(
        "Model loaded.", extra={"extra_fields": {"event": "model_loading_complete"}}
    )
    yield
    # Ничего не освобождаем: модель реюзается до остановки процесса


app = FastAPI(title="NER Service", version="1.0.0", lifespan=lifespan)

# Разрешаем публичный доступ (для бота и внешнего URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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
    start_time = time.time()

    if model is None:
        logger.error(
            "Model not ready",
            extra={
                "extra_fields": {"endpoint": "/api/predict", "execution_time_ms": 0}
            },
        )
        raise HTTPException(status_code=503, detail="Model not ready")

    text = (body.input or "").strip()

    if not text:
        execution_time = (time.time() - start_time) * 1000

        logger.info(
            "Empty input, returning empty result",
            extra={
                "extra_fields": {
                    "endpoint": "/api/predict",
                    "execution_time_ms": round(execution_time, 2),
                    "result_count": 0,
                }
            },
        )
        return []

    # Жёстко ограничим длину входа (ускоряет худшие случаи)
    if len(text) > MAX_INPUT_LEN:
        text = text[:MAX_INPUT_LEN]

        logger.warning(
            "Input truncated to max length",
            extra={
                "extra_fields": {
                    "endpoint": "/api/predict",
                    "original_length": len(body.input),
                    "truncated_length": len(text),
                }
            },
        )

    # Вынесем синхронную инференс-функцию в пул потоков, чтобы не блокировать event loop
    def _infer_sync():
        return model.predict(text)

    try:
        # Гарантируем ответ < ~1с
        result = await asyncio.wait_for(
            asyncio.to_thread(_infer_sync), timeout=PREDICT_TIMEOUT
        )
        execution_time = (time.time() - start_time) * 1000

        # Log successful prediction
        logger.info(
            "Predict completed successfully",
            extra={
                "extra_fields": {
                    "endpoint": "/api/predict",
                    "execution_time_ms": round(execution_time, 2),
                    "input": text,
                    "result": result,
                }
            },
        )

        # Ожидается список словарей с ключами start_index, end_index, entity — пускаем как есть
        return result
    except asyncio.TimeoutError:
        execution_time = (time.time() - start_time) * 1000
        # Жёстко соблюдаем SLA: не висим дольше, честно отвечаем пустым списком
        logger.warning(
            "Predict timeout",
            extra={
                "extra_fields": {
                    "endpoint": "/api/predict",
                    "execution_time_ms": round(execution_time, 2),
                    "timeout_seconds": PREDICT_TIMEOUT,
                    "input": text,
                }
            },
        )
        return []
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000

        logger.error(
            "Predict failed",
            extra={
                "extra_fields": {
                    "endpoint": "/api/predict",
                    "execution_time_ms": round(execution_time, 2),
                    "input": text,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            },
        )
        # Для стабильности сервиса — 200 с пустым списком (можно поменять на 500)
        return []


@app.get("/")
async def root():
    return {"service": "NER", "docs": "/docs"}
