import torch
import torch.nn as nn

from .layers import WaveNet


class Flow(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, num_layers, n_flows):
        super(Flow, self).__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(AffineCoupling(in_channels, channels, kernel_size, num_layers))
            self.flows.append(Flip())

    def forward(self, z, z_mask):
        for flow in self.flows:
            z = flow(z, z_mask)
        return z

    def backward(self, y, y_mask):
        for flow in reversed(self.flows):
            y = flow(y, y_mask)
        return y

    def remove_weight_norm(self):
        for flow in self.flows:
            flow.remove_weight_norm()


class Flip(nn.Module):
    def forward(self, z, z_mask, **kwargs):
        z = torch.flip(z, [1])
        return z

    def backward(self, y, y_mask, **kwargs):
        return y

    def remove_weight_norm(self):
        pass


class AffineCoupling(nn.Module):

    def __init__(self, in_channels, channels, kernel_size, num_layers, gin_channels=0, dropout=0.05):
        super(AffineCoupling, self).__init__()

        self.split_channels = in_channels // 2

        self.start = torch.nn.utils.weight_norm(nn.Conv1d(in_channels // 2, channels, 1))
        self.net = WaveNet(channels, kernel_size, num_layers, gin_channels=gin_channels, dropout=dropout)
        self.end = nn.Conv1d(channels, in_channels, 1)
        self.end.weight.data.zero_()
        self.end.bias.data.zero_()

    def forward(self, z, z_mask, g=None):
        z0, z1 = self.squeeze(z)
        z0, z1 = self._transform(z0, z1, z_mask, g=g)
        z = self.unsqueeze(z0, z1)
        return z

    def backward(self, y, y_mask, g=None):
        y0, y1 = self.squeeze(y)
        y0, y1 = self._inverse_transform(y0, y1, y_mask, g=g)
        y = self.unsqueeze(y0, y1)
        return y

    def _transform(self, z0, z1, z_mask, g):
        params = self.start(z1) * z_mask
        params = self.net(params, z_mask, g=g)
        t = self.end(params) * z_mask

        z0 = z0 + t
        z0 *= z_mask

        return z0, z1

    def _inverse_transform(self, y0, y1, y_mask, g):
        params = self.start(y1) * y_mask
        params = self.net(params, y_mask, g=g)
        t = self.end(params) * y_mask

        y0 = y0 - t
        y0 *= y_mask

        return y0, y1

    @staticmethod
    def squeeze(z, dim=1):
        C = z.size(dim)
        z0, z1 = torch.split(z, C // 2, dim=dim)
        return z0, z1

    @staticmethod
    def unsqueeze(z0, z1, dim=1):
        z = torch.cat([z0, z1], dim=dim)
        return z

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.start)
        self.net.remove_weight_norm()
