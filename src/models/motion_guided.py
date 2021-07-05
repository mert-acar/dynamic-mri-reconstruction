import torch
from torch import nn
from math import ceil
import torch.fft as fft
import torch.nn.functional as F
from data_consistency import DataConsistencyInKspace


class UNetConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, pad=False, batch_norm=False):
        super(UNetConvBlock, self).__init__()
        block = []
        padding = 0
        if pad:
            padding = kernel_size // 2
        block.append(nn.Conv2d(in_feat, out_feat, kernel_size, padding=padding))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_feat))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, pad=False, batch_norm=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_feat, out_feat, kernel_size=2, stride=2)
        self.conv_block = UNetConvBlock(
            in_feat, out_feat, kernel_size, pad, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        return self.conv_block(out)


class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size):
        super(ConvGRU, self).__init__()
        padding = kernel_size // 2
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.output = nn.Conv2d(
            hidden_size, output_size, kernel_size, padding=padding)

    def forward(self, x, prev_state):
        batch_size = x.shape[0]
        spatial_size = x.shape[2:]

        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size).to(x.device)
            # prev_state = torch.zeros(state_size)

        # data size: [batch, channel, height, width]
        stacked_inputs = torch.cat([x, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(
            torch.cat([x, prev_state * reset], dim=1)))
        new_state = prev_state * update + (1 - update) * out_inputs
        output = self.output(new_state)
        return output, new_state

class STN(nn.Module):
    def __init__(self, in_ch=128, hidden_ch=(16, 32), fc_hidden=32, transform_dims=(2,3)):
        super(STN, self).__init__()
        self.hidden_ch = hidden_ch[1]
        self.transform_dims = transform_dims
        self.localization = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch[0], kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(hidden_ch[0], hidden_ch[1], kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(hidden_ch[1] * 5 * 5, fc_hidden),
            nn.ReLU(True),
            nn.Linear(fc_hidden, transform_dims[0] * transform_dims[1])
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
       	xs = self.localization(x) 
        xs = xs.view(-1, self.hidden_ch * 5 * 5)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, self.transform_dims[0], self.transform_dims[1])

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x
	

def map_to_k_space(c, loss_mask):
    c = c[:, 0] + (1j * c[:, 1])
    c = c.permute(0, 3, 1, 2)
    k_space = fft.fftshift(fft.fft2(fft.ifftshift(c), norm='ortho'))
    k_space = loss_mask * k_space
    real = torch.unsqueeze(k_space.real, 1)
    imag = torch.unsqueeze(k_space.imag, 1)
    k_space = torch.cat((real, imag), 1).permute(0, 1, 3, 4, 2) # Since complex_mse is not implemented in torch we output this version
    return k_space


class DRN(nn.Module):
    def __init__(self, num_iter=4, depth=3, init_filter=5, hidden_dim=128, padding=True, batch_norm=False):
        super(DRN, self).__init__()
        self.N = num_iter
        kernels = [7] + [3] * (depth - 1)

        # Encoder
        self.encoder = nn.ModuleList()
        prev_ch = 2
        for i in range(depth):
            self.encoder.append(
                UNetConvBlock(prev_ch, 2 ** (init_filter + i),
                              kernels[i], pad=padding, batch_norm=batch_norm)
            )
            prev_ch = 2 ** (init_filter + i)

        # ConvGRU B - Time & Iteration
        self.B = ConvGRU(prev_ch, hidden_dim, hidden_dim, 3)

        # ConvGRU S - Iteration
        self.S = ConvGRU(prev_ch // 2, hidden_dim, prev_ch // 2, 3)

        # Decoder
        self.decoder = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.decoder.append(
                UNetUpBlock(prev_ch, 2 ** (init_filter + i),
                            kernels[i], pad=padding, batch_norm=batch_norm)
            )
            prev_ch = 2 ** (init_filter + i)

        # Where to put the ConvGRU S in the Decoding
        self.S_idx = ceil(len(self.decoder) / 2)
        
        # Project back onto image domain
        self.out = nn.Conv2d(prev_ch, 2, 1)

        # Data Consistency
        self.dc = DataConsistencyInKspace()


    def forward(self, train_image, train_k, train_mask, loss_mask=None, **args):
        '''
            The input is the undersampled image sequence with shape [N, 2, H, W, T]
            The second dimension holds the real and complex parts of the image 
            The output is the reconstructed image with the same shape
        '''
        residual = train_image.clone()
        hidden_b = None
        for t in range(train_image.shape[-1]):
            temp = residual[..., t]
            hidden_s = None
            for n in range(self.N):
                blocks = []

                # Encode
                for i, enc in enumerate(self.encoder):
                    temp = enc(temp)
                    if i != len(self.encoder) - 1:
                        blocks.append(temp)
                        temp = F.max_pool2d(temp, 2)

                temp, hidden_b = self.B(temp, hidden_b)

                # Decode
                for i, dec in enumerate(self.decoder[:self.S_idx]):
                    temp = dec(temp, blocks[-i - 1])

                temp, hidden_s = self.S(temp, hidden_s)

                for i, dec in enumerate(self.decoder[self.S_idx:]):
                    temp = dec(temp, blocks[-(i + self.S_idx) - 1])

                # Projecting in to image domain
                temp = self.out(temp)

                # Residual connection
                temp += train_image[..., t]

                # Data Consistency step
                temp = self.dc(temp, train_k[:, t, ...], train_mask[:, t, ...])

            train_image[..., t] = temp
        if loss_mask != None:
            k = map_to_k_space(train_image, loss_mask)
            return train_image, k
        else:
            return train_image


class UFlowNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, depth=5, wf=6, padding=True, batch_norm=False):
        super(UFlowNet, self).__init__()
        self.padding = padding
        self.depth = depth
        prev_ch = in_channels
        ks = 3
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_ch, 2 ** (wf + i), ks, padding, batch_norm))
            prev_ch = 2 ** (wf + i)
        
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_ch, 2 ** (wf + i), ks, padding, batch_norm))
            prev_ch = 2 ** (wf + i)
        pad = 0
        if padding:
            pad = ks // 2
        self.out = nn.Conv2d(prev_ch, out_channels, kernel_size=ks, padding=pad)


    def forward(self, x, y):
        grid = torch.cat((x, y), dim=1)
        blocks = []
        for i, down in enumerate(self.down_path):
            grid = down(grid)
            if i != len(self.down_path) - 1:
                blocks.append(grid)
                grid = F.max_pool2d(grid, 2)

        for i, up in enumerate(self.up_path):
            grid = up(grid, blocks[-i-1])

        grid = self.out(grid)
        warped = F.grid_sample(x, grid.permute(0, 2, 3, 1), align_corners=False)
        return warped, grid


class ResidualNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=2, depth=4, wf=6, padding=True, batch_norm=False):
        super(ResidualNet, self).__init__()
        self.padding = padding
        self.depth = depth
        prev_ch = in_channels
        ks = 3
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_ch, 2 ** (wf + i), ks, padding, batch_norm))
            prev_ch = 2 ** (wf + i)
        
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_ch, 2 ** (wf + i), ks, padding, batch_norm))
            prev_ch = 2 ** (wf + i)
        pad = 0
        if padding:
            pad = ks // 2
        self.out = nn.Conv2d(prev_ch, out_channels, kernel_size=ks, padding=pad)


    def forward(self, x, y, z):
        grid = torch.cat((x, y, z), dim=1)
        blocks = []
        for i, down in enumerate(self.down_path):
            grid = down(grid)
            if i != len(self.down_path) - 1:
                blocks.append(grid)
                grid = F.max_pool2d(grid, 2)

        for i, up in enumerate(self.up_path):
            grid = up(grid, blocks[-i-1])

        grid = self.out(grid)
        out = x + grid
        return out


class MODRN(nn.Module):
    def __init__(self, T=8, **kwargs):
        super(MODRN, self).__init__()
        self.drn = DRN(**kwargs['drn'])
        self.T = T
        self.me = UFlowNet(**kwargs['me'])
        self.mc = ResidualNet(**kwargs['mc'])
    
    def forward(self, train_image, train_k, train_mask, ref, loss_mask=None, **args):
        train_image = self.drn(train_image, train_k, train_mask)
        c = torch.empty(train_image.shape).to(train_image.device)
        l = 0
        r = 1
        for i in range(train_image.shape[-1]):
            z_p, _ = self.me(train_image[..., i], ref[..., l])
            z_pp, _ = self.me(train_image[..., i], ref[..., r])
            c[..., i] = self.mc(train_image[..., i], z_p, z_pp)
            if (i + 1) % self.T == 0:
                l += 1
                r += 1
        if loss_mask != None:
            k = map_to_k_space(c, loss_mask)
            return c, k
        else:
            return c


if __name__ == '__main__':
    import yaml
    with open('../config.yaml', 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    model = MODRN(**args['network'])
    inp = torch.rand(1, 2, 144, 144, 8)
    k = torch.rand(1, 8, 144, 144)
    mask = torch.rand(1, 8, 144, 144)
    ref = torch.rand(1, 2, 144, 144, 2)

    out = model(inp, k, mask, ref)
    print(out.shape)
