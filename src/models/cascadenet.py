from torch import nn
import torch.fft as fft
from .data_consistency import DataConsistencyInKspace


def map_to_k_space(c, loss_mask):
    c = c[:, 0] + (1j * c[:, 1])
    k_space = fft.fftshift(fft.fft2(fft.ifftshift(c), norm='ortho'))
    k_space = loss_mask * k_space
    real = torch.unsqueeze(k_space.real, 1)
    imag = torch.unsqueeze(k_space.imag, 1)
    k_space = torch.cat((real, imag), 1) # Since complex_mse is not implemented in torch we output this version
    return k_space


class ResnetBlock(nn.Module):
    def __init__(self, in_ch=2, num_layers=5, num_filters=64, kernel_size=3, stride=1, padding=1):
        super(ResnetBlock, self).__init__()
        self.layers = [
            nn.Conv2d(in_ch, num_filters, kernel_size, stride, padding),
            nn.ReLU()
        ]

        for i in range(1, num_layers - 1):
            self.layers.append(
                nn.Conv2d(num_filters, num_filters, kernel_size, stride, padding))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Conv2d(num_filters, in_ch,
                                     kernel_size, stride, padding))
        self.layers.append(nn.ReLU())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        residual = x.clone()
        x = self.layers(x)
        x += residual
        return x


class CascadeNetwork(nn.Module):
    def __init__(self, num_cascades=5, num_layers=5, num_filters=64, kernel_size=3, stride=1, padding=1, noise=None, **kwargs):
        super(CascadeNetwork, self).__init__()
        self.blocks = nn.ModuleList([ResnetBlock(
            2, num_layers, num_filters, kernel_size, stride, padding) for i in range(num_cascades)])
        self.dc = DataConsistencyInKspace()

    def forward(self, train_image, train_k, train_mask, loss_mask=None, **kwargs):
        for block in self.blocks:
            train_image = block(train_image)
            train_image = self.dc(train_image, train_k, train_mask)
        if loss_mask != None:
            k_space = map_to_k_space(train_image, loss_mask)
            return train_image, k_space
        return train_image


if __name__ == '__main__':
    import torch

    model = CascadeNetwork()
    inp = torch.rand(1, 2, 144, 144)
    k = torch.rand(1, 144, 144)
    mask = torch.rand(1, 144, 144)
    out = model(inp, k, mask)
    print(out.shape)
