import os

# import time
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from app.model_wrapper import ner_model

# from app.logger import logger


MAX_INPUT_LEN = int(os.getenv("MAX_INPUT_LEN", "512"))


tags_metadata = [
    {"name": "Service", "description": "Технические эндпоинты сервиса (здоровье, версия)."},
    {"name": "NER", "description": "Распознавание именованных сущностей в тексте."},
]
app = FastAPI(title="NER Service", version="1.0.0",
              description="Сервис распознавания именованных сущностей (BERT/DeepPavlov) на базе FastAPI.",
              openapi_tags=tags_metadata,
              )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthRs, tags=["Service"], summary="Проверка")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"service": "NER", "docs": "/docs"}


class PredictIn(BaseModel):
    input: str = Field("", description="Input text for NER")


class EntityOut(BaseModel):
    start_index: int
    end_index: int
    entity: str


@app.post("/api/predict", response_model=List[EntityOut],
          tags=["NER"],
          summary="Распознавание сущностей (NER)",
          description="Принимает текст и возвращает список найденных сущностей с позициями.",
          responses={
              200: {"description": "Успех: список найденных сущностей (возможно пустой)."},
              422: {"description": "Ошибка валидации входных данных."},
              500: {"description": "Внутренняя ошибка инференса."},
          },
          )
async def predict(body: PredictIn):
    # start_time = time.time()
    text = (body.input or "").strip()

    if not text:
        # logger.info("Empty input, returning empty result")
        return []

    if len(text) > MAX_INPUT_LEN:
        text = text[:MAX_INPUT_LEN]
        # logger.warning(
        #     "Input truncated to max length",
        #     extra={
        #         "extra_fields": {
        #             "endpoint": "/api/predict",
        #             "original_length": len(body.input),
        #             "truncated_length": len(text),
        #         }
        #     },
        # )

    try:
        result = ner_model.predict(text)

        # execution_time = (time.time() - start_time) * 1000
        # logger.info(
        #     "Predict completed successfully",
        #     extra={
        #         "extra_fields": {
        #             "endpoint": "/api/predict",
        #             "execution_time_ms": round(execution_time, 2),
        #             "input_sample": text,
        #             "response_payload": result,
        #         }
        #     },
        # )
        return result
    except Exception as exc:
        # execution_time = (time.time() - start_time) * 1000
        # logger.error(
        #     "Predict failed",
        #     extra={
        #         "extra_fields": {
        #             "endpoint": "/api/predict",
        #             "execution_time_ms": round(execution_time, 2),
        #             "input": text,
        #             "error": str(exc),
        #             "error_type": type(exc).__name__,
        #         }
        #     },
        # )
        return []
