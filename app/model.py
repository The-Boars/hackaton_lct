from typing import List

from deeppavlov import build_model
from deeppavlov.core.commands.utils import parse_config


class ModelPredictor:
    __slots__ = ("PROJECT_DIR", "MODEL_NAME", "ner_model")

    def __init__(self, project_dir: str = "./", model_name: str = "model"):
        self.PROJECT_DIR = project_dir
        self.MODEL_NAME = model_name

        model_config = parse_config("ner_collection3_bert")
        model_config = self.__config_change__(model_config)
        self.ner_model = build_model(model_config, download=False)

    def __config_change__(self, model_config):
        model_config["dataset_reader"]["data_path"] = self.PROJECT_DIR + "/datasets/conll/"

        del model_config['metadata']['download']

        model_config['dataset_reader']['iobes'] = False
        model_config['metadata']['variables']['MODEL_PATH'] = self.PROJECT_DIR + '/models/' + self.MODEL_NAME

        model_config['chainer']['pipe'][1]['save_path'] = self.PROJECT_DIR + '/models/tag.dict'
        model_config['chainer']['pipe'][1]['load_path'] = self.PROJECT_DIR + '/models/tag.dict'

        model_config['chainer']['pipe'][2]['save_path'] = self.PROJECT_DIR + '/models/' + self.MODEL_NAME
        model_config['chainer']['pipe'][2]['load_path'] = self.PROJECT_DIR + '/models/' + self.MODEL_NAME


        model_config['train']['batch_size'] = 400

        model_config['train']['log_every_n_batches'] = 10
        model_config['train']['val_every_n_batches'] = 10


        model_config['chainer']['pipe'][0]['in'] = ['x_tokens']
        model_config['chainer']['pipe'].insert(0, {"id": "ws_tok", "class_name": "split_tokenizer", "in": ["x"], "out": ["x_tokens"]})


        return model_config

    @staticmethod
    def _locate_token(text: str, token: str, start_pos: int) -> tuple[int, int] | None:
        if not token:
            return None
        length = len(text)
        pos = start_pos
        while pos < length and text[pos].isspace():
            pos += 1
        if text.startswith(token, pos):
            start = pos
            end = start + len(token)
            return start, end
        found = text.find(token, pos)
        if found == -1:
            found = text.find(token)
            if found == -1:
                return None
        start = found
        end = start + len(token)
        return start, end

    def _prediction(self, text: str = "") -> List[tuple[int, int, str]]:
        raw_output = self.ner_model([text])
        if not isinstance(raw_output, (list, tuple)) or len(raw_output) < 2:
            raise ValueError(f"Unexpected model output structure: {type(raw_output)}")

        tokens = raw_output[0]
        tags = raw_output[1]

        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        if tags and isinstance(tags[0], list):
            tags = tags[0]

        annotation: List[tuple[int, int, str]] = []
        cursor = 0
        for token, tag in zip(tokens, tags):
            location = self._locate_token(text, token, cursor)
            if location is None:
                continue
            start, end = location
            cursor = end
            if tag and tag != "O":
                annotation.append((start, end, tag))
        return annotation

    def _postprocess(self, text: str):
        return self._prediction(text)

    def get_response(self, text: str):
        return [
            {
                "start_index": start,
                "end_index": end,
                "entity": tag,
            }
            for start, end, tag in self._postprocess(text)
        ]
