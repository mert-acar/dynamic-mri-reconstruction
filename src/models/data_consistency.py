import torch
import torch.fft as fft


class DataConsistencyInKspace(torch.nn.Module):
    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()
        self.norm = norm
        self.noise_lvl = noise_lvl

    def forward(self, x, k0, mask):
        x = x[:, 0, ...] + (1j * x[:, 1, ...])
        k = fft.fftshift(fft.fft2(fft.ifftshift(x), norm=self.norm))

        if self.noise_lvl:
            out = (1 - mask) * k + mask * \
                (k + self.noise_lvl * k0) / (1 + self.noise_lvl)
        else:
            out = (1 - mask) * k + mask * k0

        x_res = fft.fftshift(fft.ifft2(fft.ifftshift(out), norm=self.norm))
        x_res_r = torch.unsqueeze(x_res.real, 1)
        x_res_i = torch.unsqueeze(x_res.imag, 1)
        x_res = torch.cat((x_res_r, x_res_i), 1)
        return x_res.float()
