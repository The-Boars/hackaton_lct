# NER Web‑Service (FastAPI + DeepPavlov)

Сервис для **распознавания именованных сущностей (NER)** в тексте. Реализован на **FastAPI**, использует обученную модель семейства **DeepPavlov (ner_collection3_bert)**. Эндпоинт `/api/predict` принимает текст и возвращает список найденных сущностей с позициями в исходной строке.

---

## 1) Технические требования

- **OS:** Linux / Windows / macOS (рекомендован Linux/Ubuntu 22.04 в Docker)
- **Python:** 3.10 (если без Docker)
- **Порт:** 8000 (по умолчанию)

### Опционально (GPU)
- **CUDA 11.7** и **NVIDIA Container Toolkit** (для Docker с GPU)
- Образ в `Dockerfile` рассчитан на `nvidia/cuda:11.7.1`
- Работа проверена на GPU NVIDIA A100, T4 и RTX 4070Ti Super

---

## 2) Стек и зависимости

- **FastAPI**, **Uvicorn**, **Pydantic 1.x**
- **DeepPavlov** (+ `transformers`, `tokenizers`, `torch`)
- Пользовательский JSON‑логгер (`app/logger.py`)

Список базовых пакетов — в [`requirements.txt`]. В `Dockerfile` дополнительно ставятся `fastapi`, `uvicorn` и нужная версия `torch`.

---

## 3) Быстрый старт

### 0. Подготовка модели
Необходимо скачать модель по [ссылке](https://disk.yandex.ru/d/KZluVMmhC6fP-g) и расположить ее в папке `models` 

### Вариант A — Docker (рекомендуется)

1. Установите Docker (и **NVIDIA Container Toolkit**, если хотите использовать GPU).
2. В корне проекта:
   ```bash
   docker compose up --build -d
   ```
3. Сервис поднимется на `http://localhost:8000`  
   Swagger UI: `http://localhost:8000/docs`

> В `docker-compose.yml` уже настроено пробрасывание GPU:
> ```yaml
> deploy:
>   resources:
>     reservations:
>       devices:
>         - driver: nvidia
>           count: all
>           capabilities: [gpu]
> ```

### Вариант B — Локально (без Docker)

1. Создайте окружение и установите зависимости:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   # source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   # базовые веб-зависимости (если их нет в requirements.txt)
   pip install "fastapi>=0.117.0,<1.0.0" "uvicorn[standard]>=0.29.0,<0.30.0" "pydantic>=1.8,<2.0"
   ```
   В библиотеке Deeppavlov есть конфликт с зависимостями fastapi, не влияющий на работу, приоритетная версия пакетов указана выше.
2. Установите PyTorch:
   - **CPU:** `pip install torch==1.13.1`
   - **GPU (CUDA 11.7):**
     ```bash
     pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
       --extra-index-url https://download.pytorch.org/whl/cu117
     ```
3. Подготовьте модель DeepPavlov (однократно):
   ```bash
   python -m deeppavlov install ner_collection3_bert
   ```
4. Запуск:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 3
   ```

---

## 4) Переменные окружения 

| Имя | Назначение                                                       | Значение по умолчанию |
|---|------------------------------------------------------------------|-----------------------|
| `MAX_INPUT_LEN` | Максимальная длина входной строки для `/api/predict`             | `512`                 |
| `MODEL_PROJECT_DIR` | Путь к каталогу проекта/моделей для `ModelPredictor`             | `./`                  |
| `MODEL_NAME` | Имя модели/конфига без расширений (используется в `app/model.py`) | `model`               |
| `UVICORN_WORKERS` | Количество воркеров                               | `3`                   |


Пример для Docker Compose:
```yaml
environment:
  - MAX_INPUT_LEN=512
  - MODEL_PROJECT_DIR=/app/models
  - MODEL_NAME=model
  - UVICORN_WORKERS=3
```

---

## 5) API

### 5.1 `POST /api/predict`
Распознаёт сущности в тексте.

**Request**
```json
{
  "input": "сгущенное молоко"
}
```

**Response** (пример)
```json
[
  { "start_index": 0,  "end_index": 8, "entity": "B-TYPE" },
  { "start_index": 9, "end_index": 15, "entity": "I-TYPE" }
]
```

> При ошибке обработчик возвращает пустой список `[]`, чтобы не падать на стороне клиента.

### 5.2 `GET /health`
Проверка статуса сервиса. Возвращает `200 OK` с простым JSON, например:
```json
{ "status": "ok" }
```

---

## 6) Примеры запросов

**curl**
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d "{\"input\": \"сгущенное молоко\"}"
```

---

## 7) Логирование

Используется формат JSON‑логов (`app/logger.py`):
```json
{
  "timestamp": "2025-10-01 10:36:55",
  "level": "INFO",
  "logger": "app.main",
  "message": "Predict ok",
  "extra_field": { "endpoint": "/api/predict", "execution_time_ms": 12.3 }
}
```
Логи Uvicorn access отключены, основной логгер настроен на `INFO`.

> При необходимости раскомментируйте импорт и использование `logger` в `app/main.py`.

---

## 8) Архитектура и структура проекта

```
.
├── app/
│   ├── main.py            # FastAPI-приложение (эндпоинты /health, /api/predict)
│   ├── model_wrapper.py   # Обертка над предиктором (инициализация NER-модели)
│   ├── model.py           # ModelPredictor: загрузка/вызов DeepPavlov ner_collection3_bert
│   └── logger.py          # JSON-логирование
├── models/
│   ├── model.pth.tar              # Модель
│   └── tag.dict                   # Словарь тегов
├── data_flow/          
│   ├── data_augmentation.ipynb    # Аугментация ОФД сета
│   ├── data_preparation.ipynb     # Подготовка файлов .txt для обучения модели      
│   ├── data_preprocessing.ipynb   # Разбиение на тренировочный сет и валидационный
│   ├── model_prediction.ipynb     # Предсказание моделью     
│   └── model_training.ipynb       # Обучение модели   
├── datasets/                      # Папка со всеми наборами данных 
│   └── . . .
├── requirements.txt       # Базовые зависимости (NLP / ML стек)
├── Dockerfile             # Образ на nvidia/cuda:11.7.1, подготовка модели, Uvicorn
└── docker-compose.yml     # Сервис с пробросом порта 8000 и GPU (при наличии)
```

---

## 9) Внешние источники данных и модели

- Предобученная модель **DeepPavlov `ner_collection3_bert`** (BERT‑база для NER на основе `rubert-base-cased`). Скачивается и кешируется локально через `python -m deeppavlov install ...`.
- Библиотеки **Transformers / Tokenizers / Torch** — используются для инференса.
- Во время работы веб‑сервиса внешние HTTP‑источники **не вызываются**: инференс локальный по скачанным весам.
- Обогащение данных датасетом ОФД от [Альфа-Банка + ODS](https://ods.ai/competitions/alfabank-nlp-receipts-final)
- [Матрица ошибок](https://arxiv.org/pdf/2105.05977) для моделирования опечаток

> Для полностью офлайн‑развертывания запустите установку модели в окружении с интернетом, сохраните каталоги `~/.deeppavlov` (или путь, заданный `MODEL_PROJECT_DIR`) и смонтируйте их на сервер без интернета.

---

## 10) Производительность и эксплуатация

- Запускайте `uvicorn` с несколькими воркерами (`--workers 2..4`) и выставляйте лимиты CPU/RAM.
- Для высоких нагрузок используйте **GPU** и **долгоживущие контейнеры** (прогрев модели).

---

## 11) Типичные проблемы

- **Долгая первая инференция / скачивание модели**: прогрев + загрузка весов.
- **Нет CUDA** — используйте CPU‑сборку `torch` или Docker без GPU.
- **Ошибки совместимости версий** — придерживайтесь версий из `Dockerfile`/`requirements.txt`.
- **В Windows предупреждение про `charset_normalizer`** — установите пакет `charset-normalizer` (обычно уже включён).

---


## 12) Команда

- *Канев Алексей: Project Manager, ML-Engineer, Data Scientist/ Человек*
- *Баранов Никита: Backend-разработчик*
- *Сигитов Артем: Backend-разработчик*
- *Андрианов Степан: Data Scientist*
- *Третьякова Анастасия: Data Scientist*


