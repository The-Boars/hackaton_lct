from deeppavlov import train_model, build_model 
from deeppavlov.core.commands.utils import parse_config
import re


class ModelPredictor():
    def __init__(self, project_dir: str = './', model_name: str = 'model'):
        '''
        project_dir: директория проекта, считается из файла, в котором вызывается класс
        model_name: имя модели, по которому она сохраняется в папке models, без расширения .pth.tar
        '''
        self.PROJECT_DIR = project_dir
        self.MODEL_NAME = model_name

        model_config = parse_config('ner_collection3_bert')
        model_config = self.__config_change__(model_config)

        self.ner_model = build_model(model_config, download=False)

        

    def __config_change__(self, model_config):
        model_config['dataset_reader']['data_path'] = self.PROJECT_DIR + '/datasets/conll/'

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

    def _prediction(self, input: str = ''):

        TOKEN_PATTERN = re.compile(r'\S+')

        def normalize_tag(tag: str) -> str:
            if not isinstance(tag, str):
                return 'O'
            tag = tag.strip()
            if not tag:
                return 'O'
            upper_tag = tag.upper()
            if upper_tag == 'O':
                return 'O'
            prefix = upper_tag[:2]
            if prefix == 'S-':
                return 'B-' + tag[2:]
            if prefix == 'E-':
                return 'I-' + tag[2:]
            if prefix in ('B-', 'I-'):
                return prefix + tag[2:]
            return tag

        def is_tag(value) -> bool:
            if not isinstance(value, str):
                return False
            candidate = value.strip().upper()
            if not candidate:
                return False
            if candidate == 'O':
                return True
            return len(candidate) >= 3 and candidate[1] == '-' and candidate[0] in {'B', 'I', 'S', 'E'}

        def looks_like_sequence(seq, predicate) -> bool:
            if not isinstance(seq, (list, tuple)) or not seq:
                return False
            return all(predicate(item) for item in seq)

        def looks_like_tag_sequence(seq) -> bool:
            return looks_like_sequence(seq, is_tag)

        def looks_like_token_sequence(seq) -> bool:
            return looks_like_sequence(seq, lambda item: isinstance(item, str) and not is_tag(item))

        def extract_tokens_and_tags(prediction) -> tuple[list[str], list[str]]:
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            if not isinstance(prediction, list):
                raise ValueError(f'Unexpected model output type: {type(prediction)}')
            tokens: list[str] = []
            tags: list[str] = []

            def traverse(node):
                nonlocal tokens, tags
                if isinstance(node, tuple):
                    node = list(node)
                if isinstance(node, list):
                    if not tokens and looks_like_token_sequence(node):
                        tokens = [str(item) for item in node]
                    if not tags and looks_like_tag_sequence(node):
                        tags = [normalize_tag(str(item)) for item in node]
                    if tokens and tags:
                        return
                    for child in node:
                        traverse(child)

            traverse(prediction)
            if not tags:
                raise ValueError(f'Unable to extract tag sequence from model output: {prediction}')
            return tokens, tags

        def compute_annotation(text: str, tokens: list[str], tags: list[str]) -> list[tuple[int, int, str]]:
            if not tags:
                return []
            if tokens:
                effective_len = min(len(tokens), len(tags))
                tokens = tokens[:effective_len]
                tags = tags[:effective_len]
            annotation: list[tuple[int, int, str]] = []
            if tokens:
                cursor = 0
                fallback = False
                for token, tag in zip(tokens, tags):
                    token = token or ''
                    if not tag:
                        cursor += len(token)
                        continue
                    start = text.find(token, cursor)
                    if start == -1:
                        fallback = True
                        break
                    end = start + len(token)
                    annotation.append((start, end, tag))
                    cursor = end
                if fallback:
                    annotation = []
            if not annotation:
                matches = list(TOKEN_PATTERN.finditer(text))
                effective_len = min(len(matches), len(tags))
                for match, tag in zip(matches[:effective_len], tags[:effective_len]):
                    if not tag:
                        continue
                    annotation.append((match.start(), match.end(), tag))
            return annotation

        model_output = self.ner_model([input])
        tokens, tags = extract_tokens_and_tags(model_output)
        annotation = compute_annotation(input, tokens, tags)
        return annotation
    
    def _postprocess(self, prediction):
        return self._prediction(prediction)


    def get_response(self, prediction):
        postprocessed = self._postprocess(prediction)
        response = []
        for word in postprocessed:
            start, end, tag = word
            response.append({
                'start_index': start,
                'end_index': end,
                'entity': tag
            })
        return response
