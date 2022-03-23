import torch
import torch.nn as nn

from .conformer import Conformer
from .flow import Flow
from .gan import Generator
from .layers import EmbeddingLayer, RelPositionalEncoding, PosteriorEncoder
from .loss import kl_loss
from .predictors import VarianceAdopter
from .utils import sequence_mask, generate_path, rand_slice_segments


class TTSModel(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.segment_size = params.mel_segment

        self.emb = EmbeddingLayer(**params.embedding)
        self.relative_pos_emb = RelPositionalEncoding(
            params.encoder.channels,
            params.encoder.dropout
        )
        self.encoder = Conformer(**params.encoder)
        self.variance_adopter = VarianceAdopter(**params.variance_adopter)
        self.stat_proj = nn.Conv1d(params.encoder.channels, params.encoder.channels * 2, 1)
        self.flow = Flow(**params.flow)

        self.posterior_encoder = PosteriorEncoder(**params.posterior_encoder)

        self.generator = Generator(**params.generator)

    def forward(self, inputs):
        *labels, x_length = inputs
        x = self.emb(*labels)
        x, pos_emb = self.relative_pos_emb(x)
        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)

        x = self.encoder(x, pos_emb, x_mask)
        x, y_mask = self.variance_adopter.infer(x, x_mask)
        x, pos_emb = self.relative_pos_emb(x)
        x = self.decoder(x, pos_emb, y_mask)
        x = self.out_conv(x)
        x *= y_mask
        o = self.generator(x)
        return o, x

    def compute_loss(self, batch):
        (
            *labels,
            x_length,
            _,
            spec,
            _,
            y_length,
            duration,
            pitch,
            energy
        ) = batch
        x = self.emb(*labels)
        x, pos_emb = self.relative_pos_emb(x)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        y_mask = sequence_mask(y_length).unsqueeze(1).to(x.dtype)

        x = self.encoder(x, pos_emb, x_mask)
        x = self.stat_proj(x) * x_mask
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        path = generate_path(duration.squeeze(1), attn_mask.squeeze(1))
        x, (dur_pred, pitch_pred, energy_pred) = self.variance_adopter(
            x,
            x_mask,
            y_mask,
            pitch,
            energy,
            path
        )
        m_p, logs_p = torch.chunk(x, 2, dim=1)
        z, mu_q, logs_q = self.posterior_encoder(spec, y_mask)

        z_p = self.flow(z, y_mask)

        _kl_loss = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
        duration_mask = (duration != 0).float()
        duration_loss = ((dur_pred - duration.add(1e-5).log()) * duration_mask).pow(2).sum() / torch.sum(x_length)
        pitch_loss = (pitch_pred - pitch).pow(2).sum() / torch.sum(y_length)
        energy_loss = (energy_pred - energy).pow(2).sum() / torch.sum(y_length)

        loss = _kl_loss + duration_loss + pitch_loss + energy_loss

        loss_dict = dict(
            loss=loss,
            kl_loss=_kl_loss,
            duration=duration_loss,
            pitch=pitch_loss,
            energy=energy_loss
        )

        z_slice, ids_slice = rand_slice_segments(z_p, y_length, self.segment_size)
        o = self.generator(z_slice)

        return o, ids_slice, loss_dict

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        super(TTSModel, self).load_state_dict(state_dict, strict)
        self.flow.remove_weight_norm()
        self.posterior_encoder.remove_weight_norm()
        self.generator.remove_weight_norm()
