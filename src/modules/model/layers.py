import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from .utils import get_padding


LRELU_SLOPE = 0.1


class TTSLearnEmbeddingLayer(nn.Module):
    def __init__(self, n_phoneme, channels):
        super(TTSLearnEmbeddingLayer, self).__init__()
        self.scale = math.sqrt(channels)

        self.emb = nn.Embedding(n_phoneme, channels)
        nn.init.normal_(self.emb.weight, 0.0, channels ** -0.5)

    def forward(self, x):
        x = self.emb(x) * self.scale
        x = x.transpose(-1, -2)
        return x


class PAFEmbeddingLayer(nn.Module):
    def __init__(self, n_p, n_a, n_f, channels):
        super(PAFEmbeddingLayer, self).__init__()
        self.scale = math.sqrt(channels)

        self.p_emb = nn.Embedding(n_p, channels)
        nn.init.normal_(self.p_emb.weight, 0.0, channels ** -0.5)

        self.a_emb = nn.Embedding(n_a, channels)
        nn.init.normal_(self.a_emb.weight, 0.0, channels ** -0.5)

        self.f_emb = nn.Embedding(n_f, channels)
        nn.init.normal_(self.f_emb.weight, 0.0, channels ** -0.5)

    def forward(self, p, a, f):
        p = self.p_emb(p) * self.scale
        a = self.a_emb(a) * self.scale
        f = self.f_emb(f) * self.scale
        x = torch.cat([p, a, f], dim=-1).transpose(-1, -2)
        return x


class PPAddEmbeddingLayer(nn.Module):
    def __init__(self, n_phoneme, n_prosody, channels):
        super(PPAddEmbeddingLayer, self).__init__()
        self.scale = math.sqrt(channels)

        self.phoneme_emb = nn.Embedding(n_phoneme, channels)
        nn.init.normal_(self.phoneme_emb.weight, 0.0, channels ** -0.5)

        self.prosody_emb = nn.Embedding(n_prosody, channels)
        nn.init.normal_(self.prosody_emb.weight, 0.0, channels ** -0.5)

    def forward(self, phoneme, prosody):
        phoneme = self.phoneme_emb(phoneme) * self.scale
        prosody = self.prosody_emb(prosody) * self.scale
        x = (phoneme + prosody).transpose(-1, -2)
        return x


class EmbeddingLayer(nn.Module):
    _d = {
        'ttslearn': TTSLearnEmbeddingLayer,
        'paf': PAFEmbeddingLayer,
        'pp_add': PPAddEmbeddingLayer
    }

    def __init__(self, mode, **kwargs):
        super(EmbeddingLayer, self).__init__()
        self.emb = self._d[mode](**kwargs)

    def forward(self, *args, **kwargs):
        return self.emb(*args, **kwargs)


class RelPositionalEncoding(nn.Module):
    def __init__(self, channels, dropout=0.1, max_len=10000):
        super(RelPositionalEncoding, self).__init__()
        self.d_model = channels
        self.scale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(2) >= x.size(2) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.transpose(-1, -2).to(device=x.device, dtype=x.dtype)

    def forward(self, x):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            :,
            self.pe.size(2) // 2 - x.size(2) + 1 : self.pe.size(2) // 2 + x.size(2),
        ]
        return x, self.dropout(pos_emb)


class FFN(nn.Module):
    def __init__(self, channels, dropout):
        super(FFN, self).__init__()

        self.norm = LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels * 4, 1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(channels * 4, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x * x_mask)
        x = self.dropout(x)
        return x * x_mask


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        x = x * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
        return x


class WaveNet(nn.Module):
    def __init__(self, channels, kernel_size, num_layers, dilation_rate=1, gin_channels=0, dropout=0):
        super(WaveNet, self).__init__()

        self.channels = channels
        self.num_layers = num_layers

        self.dilated_convs = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            conv = nn.Conv1d(channels, channels * 2, kernel_size, padding=padding, dilation=dilation)
            conv = nn.utils.weight_norm(conv)
            self.dilated_convs.append(conv)

        self.out_convs = nn.ModuleList()
        for i in range(num_layers):
            conv = nn.Conv1d(channels, channels * 2 if i < num_layers-1 else channels, 1)
            conv = nn.utils.weight_norm(conv)
            self.out_convs.append(conv)

        self.dropout = nn.Dropout(dropout)

        if gin_channels > 0:
            self.cond_layer = nn.Conv1d(gin_channels, channels, 1)

    def forward(self, x, x_mask, g=None):
        if g is not None:
            g = self.cond_layer(g)
        out = 0
        for i, (d_conv, o_conv) in enumerate(zip(self.dilated_convs, self.out_convs)):
            x_in = d_conv(x)
            if g is not None:
                x_in += g
            x_in_a, x_in_b = x_in.chunk(2, dim=1)
            x_in = x_in_a.sigmoid() * x_in_b.tanh()
            if i < self.num_layers - 1:
                o1, o2 = o_conv(x_in).chunk(2, dim=1)
                x = (x + o1) * x_mask
                x = self.dropout(x)
                out += o2 * x_mask
            else:
                out += o_conv(x_in)
        return out * x_mask

    def remove_weight_norm(self):
        for l in self.dilated_convs:
            nn.utils.remove_weight_norm(l)
        for l in self.out_convs:
            nn.utils.remove_weight_norm(l)


class ConvolutionModule(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super(ConvolutionModule, self).__init__()
        self.layer_norm = LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels * 2, 1)
        self.glu = GLU(dim=1)
        self.depth_wise_conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.batch_norm = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x = self.layer_norm(x)
        x = self.conv1(x) * x_mask
        x = self.glu(x)
        x = self.depth_wise_conv(x) * x_mask
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.conv2(x) * x_mask
        x = self.dropout(x)
        return x


class GLU(nn.Module):
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.glu(x, self.dim)


class RelativeSelfAttentionLayer(nn.Module):
    def __init__(self, channels, n_heads, dropout):
        super(RelativeSelfAttentionLayer, self).__init__()
        self.norm = LayerNorm(channels)
        self.mha = RelativeMultiHeadAttention(channels, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = self.norm(x)
        x = self.mha(x, x, x, pos_emb, attn_mask)
        x = self.dropout(x)
        return x


class RelativeMultiHeadAttention(nn.Module):

    def __init__(self, channels, num_heads, dropout):
        super(RelativeMultiHeadAttention, self).__init__()
        assert channels % num_heads == 0, "d_model % num_heads should be zero."
        self.channels = channels
        self.inner_channels = channels // num_heads
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(channels)

        self.query_proj = nn.Conv1d(channels, channels, 1)
        self.key_proj = nn.Conv1d(channels, channels, 1)
        self.value_proj = nn.Conv1d(channels, channels, 1)
        self.pos_proj = nn.Conv1d(channels, channels, 1, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.inner_channels))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.inner_channels))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Conv1d(channels, channels, 1)

    def forward(self, q, k, v, pos_emb, mask=None):
        B = q.size(0)

        q = self.query_proj(q).view(B, self.num_heads, self.inner_channels, -1)
        k = self.key_proj(k).view(B, self.num_heads, self.inner_channels, -1)
        v = self.value_proj(v).view(B, self.num_heads, self.inner_channels, -1)

        B_pos = pos_emb.size(0)
        pos_emb = self.pos_proj(pos_emb).view(B_pos, self.num_heads, self.inner_channels, -1)

        content_score = torch.matmul((q + self.u_bias[None, :, :, None]).transpose(-1, -2), k)
        pos_score = torch.matmul((q + self.v_bias[None, :, :, None]).transpose(-1, -2), pos_emb)
        pos_score = self.rel_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e4)

        attn_map = F.softmax(score, -1)
        attn = self.dropout(attn_map)

        context = torch.matmul(v, attn)
        context = context.contiguous().view(B, self.channels, -1)

        return self.out_proj(context)

    @staticmethod
    def rel_shift(x):
        B, H, T1, T2 = x.size()
        zero_pad = torch.zeros((B, H, T1, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(B, H, T2 + 1, T1)
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, :T2 // 2 + 1
        ]
        return x


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                  padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                                  padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                                  padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(self.init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(self.init_weights)

    @staticmethod
    def init_weights(m, mean=0.0, std=0.01):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(mean, std)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation_rate=1, num_layers=16):
        super(PosteriorEncoder, self).__init__()
        self.in_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.conv = WaveNet(out_channels, kernel_size, num_layers, dilation_rate)
        self.out_conv = nn.Conv1d(out_channels, out_channels * 2, 1)

    def forward(self, x, x_mask):
        x = self.in_conv(x) * x_mask
        x = self.conv(x, x_mask)
        x = self.out_conv(x) * x_mask
        m, logs = torch.chunk(x, 2, dim=1)
        z = (m + torch.exp(logs) * torch.randn_like(m)) * x_mask
        return z, m, logs

    def remove_weight_norm(self):
        self.conv.remove_weight_norm()
