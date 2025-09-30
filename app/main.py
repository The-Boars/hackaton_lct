import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.model_wrapper import NERModel


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
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        return json.dumps(log_entry, ensure_ascii=False)


logging.basicConfig(level=logging.INFO)
root_logger = logging.getLogger()
root_logger.handlers.clear()
console_handler = logging.StreamHandler()
console_handler.setFormatter(JSONFormatter())
root_logger.addHandler(console_handler)
logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


PREDICT_TIMEOUT = float(os.getenv("PREDICT_TIMEOUT_SEC", "0.95"))
MAX_INPUT_LEN = int(os.getenv("MAX_INPUT_LEN", "512"))
DEFAULT_MODEL_POOL = max(1, os.cpu_count() or 1)
_model_pool_env = os.getenv("MODEL_POOL_SIZE")
if _model_pool_env is not None:
    try:
        MODEL_POOL_SIZE = max(1, int(_model_pool_env))
    except ValueError:
        logger.warning(
            "Invalid MODEL_POOL_SIZE value '%s', falling back to %s",
            _model_pool_env,
            DEFAULT_MODEL_POOL,
        )
        MODEL_POOL_SIZE = DEFAULT_MODEL_POOL
else:
    MODEL_POOL_SIZE = DEFAULT_MODEL_POOL

DEFAULT_EXECUTOR_WORKERS = MODEL_POOL_SIZE * 2
_executor_env = os.getenv("PREDICT_POOL_SIZE")
if _executor_env is not None:
    try:
        PREDICT_POOL_SIZE = max(1, int(_executor_env))
    except ValueError:
        logger.warning(
            "Invalid PREDICT_POOL_SIZE value '%s', falling back to %s",
            _executor_env,
            DEFAULT_EXECUTOR_WORKERS,
        )
        PREDICT_POOL_SIZE = DEFAULT_EXECUTOR_WORKERS
else:
    PREDICT_POOL_SIZE = DEFAULT_EXECUTOR_WORKERS

MAX_LOG_INPUT_LEN = int(os.getenv("MAX_LOG_INPUT_LEN", "160"))

model_queue: asyncio.Queue[NERModel] | None = None
predict_executor: ThreadPoolExecutor | None = None


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_queue
    global predict_executor

    logger.info(
        "Loading NER models",
        extra={
            "extra_fields": {
                "event": "model_loading_start",
                "instances": MODEL_POOL_SIZE,
            }
        },
    )
    model_queue = asyncio.Queue(maxsize=MODEL_POOL_SIZE)
    for index in range(MODEL_POOL_SIZE):
        model_instance = NERModel()
        await model_queue.put(model_instance)
        logger.info(
            "Model instance ready",
            extra={
                "extra_fields": {
                    "event": "model_instance_ready",
                    "index": index,
                }
            },
        )

    predict_executor = ThreadPoolExecutor(
        max_workers=PREDICT_POOL_SIZE,
        thread_name_prefix="predict",
    )
    loop = asyncio.get_running_loop()
    loop.set_default_executor(predict_executor)
    logger.info(
        "Predict executor created",
        extra={
            "extra_fields": {
                "event": "executor_ready",
                "max_workers": PREDICT_POOL_SIZE,
            }
        },
    )

    try:
        yield
    finally:
        if predict_executor is not None:
            predict_executor.shutdown(wait=True)
            logger.info(
                "Predict executor shut down",
                extra={"extra_fields": {"event": "executor_shutdown"}},
            )
        if model_queue is not None:
            while not model_queue.empty():
                await model_queue.get()
                model_queue.task_done()
            model_queue = None


app = FastAPI(title="NER Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictIn(BaseModel):
    input: str = Field("", description="Input text for NER")


class EntityOut(BaseModel):
    start_index: int
    end_index: int
    entity: str


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _acquire_model() -> NERModel:
    if model_queue is None:
        raise HTTPException(status_code=503, detail="Model pool not ready")
    return await model_queue.get()


async def _release_model(instance: NERModel) -> None:
    if model_queue is None:
        return
    await model_queue.put(instance)


@app.post("/api/predict", response_model=List[EntityOut])
async def predict(body: PredictIn):
    if model_queue is None or predict_executor is None:
        logger.error(
            "Predict attempted before service ready",
            extra={"extra_fields": {"endpoint": "/api/predict"}},
        )
        raise HTTPException(status_code=503, detail="Model is not ready")

    start_time = time.time()
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

    model_instance = await _acquire_model()

    def _infer_sync() -> List[dict]:
        return model_instance.predict(text)

    loop = asyncio.get_running_loop()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(predict_executor, _infer_sync),
            timeout=PREDICT_TIMEOUT,
        )
        execution_time = (time.time() - start_time) * 1000
        logger.info(
            "Predict completed successfully",
            extra={
                "extra_fields": {
                    "endpoint": "/api/predict",
                    "execution_time_ms": round(execution_time, 2),
                    "input_length": len(text),
                    "result_count": len(result),
                    "input_sample": text,
                    "response_payload": result,
                    "input_preview": _shorten(text, MAX_LOG_INPUT_LEN),
                }
            },
        )
        return result
    except asyncio.TimeoutError:
        execution_time = (time.time() - start_time) * 1000
        logger.warning(
            "Predict timeout",
            extra={
                "extra_fields": {
                    "endpoint": "/api/predict",
                    "execution_time_ms": round(execution_time, 2),
                    "timeout_seconds": PREDICT_TIMEOUT,
                    "input_length": len(text),
                }
            },
        )
        return []
    except Exception as exc:
        execution_time = (time.time() - start_time) * 1000
        logger.error(
            "Predict failed",
            extra={
                "extra_fields": {
                    "endpoint": "/api/predict",
                    "execution_time_ms": round(execution_time, 2),
                    "input_length": len(text),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            },
        )
        return []
    finally:
        await _release_model(model_instance)


@app.get("/")
async def root():
    return {"service": "NER", "docs": "/docs"}

