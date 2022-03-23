import numpy as np
from abc import abstractmethod


class TokenizerBase:

    def __init__(self, config):
        self.config = config

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    @abstractmethod
    def tokenize(self, *args, **kwargs):
        pass

    @abstractmethod
    def extract(self, *args, **kwargs):
        pass

    def extract_duration(self, label, sr, y_length):
        duration = list()
        for b, e, _ in label[1:-1]:
            d = (e - b) * 1e-7 * sr / 256
            duration += [d]
        duration = self.refine_duration(duration, y_length)
        return duration

    @staticmethod
    def refine_duration(duration, y_length):
        duration = np.array(duration)
        duration_floor = np.floor(duration)
        diff_rest = y_length - np.sum(duration_floor)
        indices = np.argsort(np.abs(duration - duration_floor))
        for idx in indices:
            duration_floor[idx] += 1
            diff_rest -= 1
            if diff_rest == 0:
                break
        return duration_floor
