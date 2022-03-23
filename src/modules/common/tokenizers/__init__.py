from .ttslearn import TTSLearnTokenizer
from .paf import PAFTokenizer
from .pp_add import PPAddTokenizer


class Tokenizer:
    _d = {
        'ttslearn': TTSLearnTokenizer,
        'paf': PAFTokenizer,
        'pp_add': PPAddTokenizer
    }

    @classmethod
    def from_config(cls, config):
        return cls._d[config.mode](config)
