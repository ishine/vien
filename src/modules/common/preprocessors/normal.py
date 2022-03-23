from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from .base import PreProcessorBase
from ..transforms import Spectrogram, MelSpectrogramWithEnergy
from ..tokenizers import Tokenizer

ORIG_SR = None
NEW_SR = None


class NormalPreProcessor(PreProcessorBase):

    def __init__(self, config):
        self.wav_dir = Path(config.wav_dir)
        self.label_dir = Path(config.label_dir)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.to_spec = Spectrogram(**config.spec)
        self.to_mel = MelSpectrogramWithEnergy(**config.mel)

        self.tokenizer = Tokenizer.from_config(config.tokenizer)

        global ORIG_SR, NEW_SR
        ORIG_SR = config.orig_sr
        NEW_SR = config.new_sr

    @staticmethod
    def get_time(label_path, sr=48000):
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        b, e = lines[1], lines[-2]
        begin_time = int(int(b.split(' ')[0]) * 1e-7 * sr)
        end_time = int(int(e.split(' ')[1]) * 1e-7 * sr)
        return begin_time, end_time

    def load_wav(self, wav_path, label_path):
        wav, sr = sf.read(wav_path)
        assert sr == ORIG_SR
        b, e = self.get_time(label_path, sr=ORIG_SR)
        wav = wav[b:e]
        wav = librosa.resample(wav, ORIG_SR, NEW_SR)
        return wav

    @staticmethod
    def extract_feats(wav):
        f0, sp, ap = pw.wav2world(wav, NEW_SR, 1024, 256 / NEW_SR * 1000)
        return f0, sp, ap

    def process_speaker(self, wav_dir_path, label_dir_path):
        wav_paths = list(sorted(wav_dir_path.glob('*.wav')))
        label_paths = list(sorted(label_dir_path.glob('*.lab')))

        s_p = StandardScaler()
        s_e = StandardScaler()

        for i in tqdm(range(len(wav_paths))):
            wav = self.load_wav(wav_paths[i], label_paths[i])
            pitch, *_ = self.extract_feats(wav)
            wav = torch.FloatTensor(wav).view(1, -1)
            spec = self.to_spec(wav)
            spec = spec.squeeze()
            mel, energy = self.to_mel(wav)
            mel, energy = mel.squeeze(), energy.squeeze()
            pitch = pitch[:mel.size(-1)]
            label, duration = self.tokenizer.extract(label_paths[i], NEW_SR, mel.size(-1))

            assert sum(duration) == mel.size(-1), f'{sum(duration)}, {mel.size(-1)}'

            pitch = np.array(pitch).astype(np.float32)
            energy = np.array(energy).astype(np.float32)

            pitch[pitch != 0] = np.log(pitch[pitch != 0])
            energy[energy != 0] = np.log(energy[energy != 0])

            assert pitch.shape[0] == mel.size(-1)
            assert energy.shape[0] == mel.size(-1)
            assert mel.size(-1) == spec.size(-1)

            s_p.partial_fit(pitch.reshape(-1, 1))
            s_e.partial_fit(energy.reshape(-1, 1))

            if i < 3:
                print(wav.size())
                print(mel.size())
                print(label)
                print(duration)
                print(pitch.shape)
                print(energy.shape)

            torch.save([
                wav,
                spec,
                mel,
                label,
                torch.LongTensor(duration).view(1, -1),
                torch.FloatTensor(pitch).view(1, -1),
                torch.FloatTensor(energy).view(1, -1)
            ], self.output_dir / f'data_{i+1:04d}.pt')

        torch.save({
            'pitch_mean': s_p.mean_,
            'pitch_std': np.sqrt(s_p.var_),
            'energy_mean': s_e.mean_,
            'energy_std': np.sqrt(s_e.var_)
        }, self.output_dir / f'stat.pt')

    def run(self):
        print('Start Preprocessing')
        self.process_speaker(self.wav_dir, self.label_dir)

