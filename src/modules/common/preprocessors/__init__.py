from .normal import NormalPreProcessor


class PreProcessor:
    _d = {
        'normal': NormalPreProcessor
    }

    @classmethod
    def from_config(cls, config):
        return cls._d[config.preprocess.mode](config.preprocess)
