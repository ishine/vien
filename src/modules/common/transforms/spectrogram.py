import torch
import torchaudio


class Spectrogram(torchaudio.transforms.Spectrogram):
    def forward(self, waveform):
        spec = super(Spectrogram, self).forward(waveform)
        spec = torch.log(torch.clamp(spec, min=1e-5))
        return spec


class MelSpectrogramWithEnergy(torchaudio.transforms.MelSpectrogram):
    def forward(self, waveform):
        spec = self.spectrogram(waveform)
        energy = torch.norm(spec, dim=1)
        mel = self.mel_scale(spec)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel, energy


class MelSpectrogram(MelSpectrogramWithEnergy):
    def forward(self, waveform):
        return super(MelSpectrogram, self).forward(waveform)[0]
