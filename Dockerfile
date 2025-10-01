FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update && apt install -y python3.10 && apt-get install -y pip


# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall "fastapi>=0.117.0,<0.120.0" "uvicorn[standard]>=0.29.0,<0.30.0" "pydantic>=1.8.0,<2.0.0"

RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

RUN python3 -m deeppavlov install ner_collection3_bert

# Copy application code
COPY ./models ./models
COPY ./app ./app
# RUN python3 -c "from app.model import ModelPredictor; print('Pre-warming model...'); m = ModelPredictor(); print('Pre-warm completed')"

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
