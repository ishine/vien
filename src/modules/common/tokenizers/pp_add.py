import re
import torch
from nnmnkwii.io import hts
from ttslearn.tacotron.frontend.openjtalk import pp_symbols, numeric_feature_by_regex
from ttslearn.tacotron.frontend.openjtalk import phonemes, extra_symbols, num_vocab
from .base import TokenizerBase


class PPAddTokenizer(TokenizerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phoneme_dict = {s: i for i, s in enumerate(['<pad>'] + phonemes)}
        self.prosody_dict = {s: i for i, s in enumerate(['<pad>'] + extra_symbols)}

    def tokenize(self, inputs):
        phoneme, prosody = inputs
        phoneme = [self.phoneme_dict[s] for s in phoneme]
        prosody = [self.prosody_dict[s] for s in prosody]
        is_transpose = [0, 0]
        return torch.LongTensor(phoneme), torch.LongTensor(prosody), is_transpose

    def __len__(self):
        return len(self.phoneme_dict)

    def extract(self, label_path, sr, y_length):
        label = hts.load(label_path)
        phoneme, prosody = self.pp_symbols(label.contexts)
        assert len(phoneme) == len(prosody), f'\n{pp_symbols(label.contexts)}\n{phoneme}\n{prosody}\n{len(phoneme)}, {len(prosody)}'

        duration = self.extract_duration(label, sr, y_length)
        return (phoneme, prosody), duration

    @staticmethod
    def pp_symbols(labels, drop_unvoiced_vowels=True):
        phoneme, prosody = list(), list()
        N = len(labels)
        # 各音素毎に順番に処理
        flag = False
        for n in range(N):
            lab_curr = labels[n]

            # 当該音素
            p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)  # type: ignore

            # 無声化母音を通常の母音として扱う
            if drop_unvoiced_vowels and p3 in "AEIOU":
                p3 = p3.lower()
            # 先頭と末尾の sil のみ例外対応
            if p3 == "sil":
                assert n == 0 or n == N - 1
                if n == 0:
                    prosody.append("^")
                elif n == N - 1:
                    # 疑問系かどうか
                    e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                    if e3 == 0:
                        prosody[-1] = "$"
                    elif e3 == 1:
                        prosody[-1] = "?"
                continue
            else:
                phoneme.append(p3)

            if flag:
                flag = False
                continue

            # アクセント型および位置情報（前方または後方）
            a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
            a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
            a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
            # アクセント句におけるモーラ数
            f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

            a2_next = numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])

            # アクセント句境界
            if a3 == 1 and a2_next == 1:
                prosody.append("#")
            # ピッチの立ち下がり（アクセント核）
            elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
                prosody.append("]")
            # ピッチの立ち上がり
            elif a2 == 1 and a2_next == 2:
                prosody.append("[")
            else:
                if n != 1:
                    prosody.append('_')

            if len(phoneme) != len(prosody):
                flag = True

        return phoneme, prosody
