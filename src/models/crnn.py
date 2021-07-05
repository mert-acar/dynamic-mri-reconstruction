import torch
import torch.nn as nn
import torch.fft as fft
from data_consistency import DataConsistencyInKspace

def map_to_k_space(c, loss_mask):
    c = c[:, 0] + (1j * c[:, 1])
    c = c.permute(0, 3, 1, 2)
    k_space = fft.fftshift(fft.fft2(fft.ifftshift(c), norm='ortho'))
    k_space = loss_mask * k_space
    real = torch.unsqueeze(k_space.real, 1)
    imag = torch.unsqueeze(k_space.imag, 1)
    k_space = torch.cat((real, imag), 1).permute(0, 1, 3, 4, 2) # Since complex_mse is not implemented in torch we output this version
    return k_space

class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations
    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_iteration, hidden):
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        ih_to_ih = self.ih2ih(hidden_iteration)
        hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)
        return hidden


class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer
    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode
    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)

    def forward(self, input, input_iteration, test=False):
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            with torch.no_grad():
                hid_init = torch.zeros(size_h).to(input.device)
        else:
            hid_init = torch.zeros(size_h).to(input.device)

        output_f = []
        output_b = []
        # forward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden)
            output_f.append(hidden)

        output_f = torch.cat(output_f)

        # backward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[nt - i - 1], input_iteration[nt - i -1], hidden)
            output_b.append(hidden)

        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output


class CRNN(nn.Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks
    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image, kspace_data, mask], each of shape (batch_size, 2, width, height, n_seq)
    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, 2, width, height, n_seq)
    """
    def __init__(self, num_filters=64, kernel_size=3, num_iterations=5, num_layers=5, **kwargs):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNN, self).__init__()
        self.nc = num_iterations
        self.nd = num_layers
        self.nf = num_filters
        self.ks = kernel_size

        self.bcrnn = BCRNNlayer(2, self.nf, self.ks)
        self.conv1_x = nn.Conv2d(self.nf, self.nf, self.ks, padding = self.ks//2)
        self.conv1_h = nn.Conv2d(self.nf, self.nf, self.ks, padding = self.ks//2)
        self.conv2_x = nn.Conv2d(self.nf, self.nf, self.ks, padding = self.ks//2)
        self.conv2_h = nn.Conv2d(self.nf, self.nf, self.ks, padding = self.ks//2)
        self.conv3_x = nn.Conv2d(self.nf, self.nf, self.ks, padding = self.ks//2)
        self.conv3_h = nn.Conv2d(self.nf, self.nf, self.ks, padding = self.ks//2)
        self.conv4_x = nn.Conv2d(self.nf, 2, self.ks, padding = self.ks//2)
        self.relu = nn.ReLU(inplace=True)

        self.dcs = nn.ModuleList()
        for i in range(self.nc):
            self.dcs.append(DataConsistencyInKspace(norm='ortho'))


    def forward(self, train_image, train_k, train_mask, loss_mask=None, test=False, **kwargs):
        """
        x   - input in image domain, of shape (n, 2, nx, ny, n_seq)
        k   - initially sampled elements in k-space
        m   - corresponding nonzero location
        test - True: the model is in test mode, False: train mode
        """
        net = {}
        n_batch, n_ch, width, height, n_seq = train_image.size()
        size_h = [n_seq*n_batch, self.nf, width, height]
        if test:
            with torch.no_grad():
                hid_init = torch.zeros(size_h).to(train_image.device)
        else:
            hid_init = torch.zeros(size_h).to(train_image.device)

        for j in range(self.nd-1):
            net['t0_x%d'%j]=hid_init

        for i in range(1,self.nc+1):
            train_image = train_image.permute(4,0,1,2,3)
            train_image = train_image.contiguous()
            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)].view(n_seq, n_batch,self.nf,width, height)
            net['t%d_x0'%i] = self.bcrnn(train_image, net['t%d_x0'%(i-1)], test)
            net['t%d_x0'%i] = net['t%d_x0'%i].view(-1,self.nf,width, height)

            net['t%d_x1'%i] = self.conv1_x(net['t%d_x0'%i])
            net['t%d_h1'%i] = self.conv1_h(net['t%d_x1'%(i-1)])
            net['t%d_x1'%i] = self.relu(net['t%d_h1'%i]+net['t%d_x1'%i])

            net['t%d_x2'%i] = self.conv2_x(net['t%d_x1'%i])
            net['t%d_h2'%i] = self.conv2_h(net['t%d_x2'%(i-1)])
            net['t%d_x2'%i] = self.relu(net['t%d_h2'%i]+net['t%d_x2'%i])

            net['t%d_x3'%i] = self.conv3_x(net['t%d_x2'%i])
            net['t%d_h3'%i] = self.conv3_h(net['t%d_x3'%(i-1)])
            net['t%d_x3'%i] = self.relu(net['t%d_h3'%i]+net['t%d_x3'%i])

            net['t%d_x4'%i] = self.conv4_x(net['t%d_x3'%i])

            train_image = train_image.view(-1, n_ch, width, height)
            net['t%d_out'%i] = train_image + net['t%d_x4'%i]

            net['t%d_out'%i] = net['t%d_out'%i].view(-1,n_batch, n_ch, width, height)
            net['t%d_out'%i] = net['t%d_out'%i].permute(1,2,0,3,4)
            net['t%d_out'%i].contiguous()
            net['t%d_out'%i] = self.dcs[i-1](net['t%d_out'%i], train_k, train_mask)
            train_image = net['t%d_out'%i].permute(0, 1, 3, 4, 2)

        net['t%d_out'%i] = net['t%d_out'%i].permute(0,1,3,4,2)
        if loss_mask != None:
            k = map_to_k_space(net['t%d_out'%i], loss_mask)
            return net['t%d_out'%i], k
        else:
            return net['t%d_out'%i]
