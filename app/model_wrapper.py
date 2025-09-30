
from typing import List, Dict, Optional
import os

# Важно: импортируем ровно так, как у тебя в ноутбуке
from app.model import ModelPredictor  # <-- твой класс

class NERModel:
    def __init__(self) -> None:
        project_dir = os.getenv("MODEL_PROJECT_DIR", "./")
        model_name = os.getenv("MODEL_NAME", "model")
        # Загружаем один раз при старте процесса
        self._predictor = ModelPredictor(project_dir=project_dir, model_name=model_name)

    def predict(self, text: str) -> List[Dict]:
        # Ожидается формат [{'start_index':..., 'end_index':..., 'entity':...}, ...]
        # Делегируем твоему коду:
        return self._predictor.get_response(text)


nermodel=NERModel()