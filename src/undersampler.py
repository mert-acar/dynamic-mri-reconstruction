import numpy as np
from utils import complex2real, fft2c, ifft2c
from numpy.lib.stride_tricks import as_strided


class Undersampler:
    def __init__(self, fold, supervision, undersampling_args):
        self.fold = fold
        self.mask_args = undersampling_args
        self.supervision = supervision

    def __call__(self, data):
        mask = self.cartesian_mask(data.shape)
        u_image, u_k = self.undersample(data, mask)

        if self.supervision == 'self':
            if self.fold == 'train':
                train_mask, loss_mask = self.ss_mask(u_k, mask)
                u_image, u_k = self.undersample(data, train_mask)
                _, ref_k = self.undersample(data, loss_mask)
            else:
                train_mask = mask
                loss_mask = np.ones_like(mask)
                ref_k = fft2c(data)

            return {
                'train_image': complex2real(u_image),
                'train_k': u_k,
                'train_mask': train_mask,
                'loss_k': complex2real(ref_k),
                'loss_mask': loss_mask
            }
            
        else:
            return {
                'train_image': complex2real(u_image),
                'train_k': u_k,
                'train_mask': mask
            }


     
    def norm(self, tensor, axes=(0, 1, 2), keepdims=True):
        """
        Parameters
        ----------
        tensor : It can be in image or k-space.
        axes :  The default is (0, 1, 2).

        Returns
        -------
        tensor : applies l2-norm .
        """
        for axis in axes:
            tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)
        if not keepdims: return tensor.squeeze()
        return tensor


    def find_center_idx(self, kspace, axes=(0, 1, 2)):
        """
        Parameters
        ----------
        kspace : ntime x nrow x ncol.
        axes :  The default is (0, 1, 2)
        Returns
        -------
        the center of the k-space
        """

        center_locs = self.norm(kspace, axes=axes).squeeze()
        return np.argsort(center_locs)[-1:]


    def normal_pdf(self, length, sensitivity):
        return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


    def undersample(self, x, mask):
        '''
        Undersample x. FFT2 will be applied to the last 2 axis

        Parameters
        ----------
        x: array_like
            data in complex image format
        mask: array_like
            undersampling mask in fourier domain
        noise_power: float
            simulates acquisition noise, complex AWG noise.
            must be percentage of the peak signal

        Returns
        -------
        xu: array_like
            undersampled image in image domain. Note that it is complex valued
        x_fu: array_like
            undersampled data in k-space
        '''

        if self.mask_args['noise_power'] != 0:
            nz = np.sqrt(.5)*(np.random.normal(0, 1, x.shape) + 1j * np.random.normal(0, 1, x.shape))
            nz = nz * np.sqrt(self.mask_args['noise_power'])
            nz = nz * np.sqrt(np.prod(mask.shape[-2:]))
        else:
            nz = 0

        x_f = fft2c(x)
        x_fu = mask * (x_f + nz)
        x_u = ifft2c(x_fu)
        
        return x_u, x_fu

    def cartesian_mask(self, shape):
        N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]

        pdf_x = self.normal_pdf(Nx, 0.5/(Nx/10.)**2)

        lmda = Nx/(2. * self.mask_args['acc'])
        n_lines = int(Nx / self.mask_args['acc'])

        pdf_x += lmda * 1./Nx

        if self.mask_args['sample_n']:
            pdf_x[Nx // 2-self.mask_args['sample_n'] // 2 : Nx // 2+self.mask_args['sample_n'] // 2] = 0
            pdf_x /= np.sum(pdf_x)
            n_lines -= self.mask_args['sample_n']

        mask = np.zeros((N, Nx))
        for i in range(N):
            idx = np.random.choice(Nx, n_lines, False, pdf_x)
            mask[i, idx] = 1

        if self.mask_args['sample_n']:
            mask[:, Nx // 2-self.mask_args['sample_n'] // 2 : Nx // 2+self.mask_args['sample_n'] // 2] = 1

        size = mask.itemsize
        mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))
        mask = mask.reshape(shape)

        return mask.astype(np.float32)


    def ss_mask(self, input_data, input_mask):
        nrow, ncol = input_data.shape[1:] 
        center_kx = int(self.find_center_idx(input_data, axes=(0, 1)))
        center_ky = int(self.find_center_idx(input_data, axes=(0, 2)))
        temp_mask = np.copy(input_mask)
        temp_mask[:, center_kx - self.mask_args['acs'][0] // 2: center_kx + self.mask_args['acs'][0] // 2,
                     center_ky - self.mask_args['acs'][1] // 2: center_ky + self.mask_args['acs'][1] // 2] = 0

        loss_mask = np.zeros_like(input_mask)
        train_mask = np.empty(input_mask.shape)
        
        for t in range(input_mask.shape[0]):
            curr_mask = input_mask[t]
            num_points = np.ceil(np.sum(curr_mask[:]) * self.mask_args['rho']).astype(int)
            count = 0
            while count <= num_points:
                indx = np.round(np.random.normal(loc=center_kx, scale=(nrow - 1) / self.mask_args['std_scale'])).astype(int)
                indy = np.round(np.random.normal(loc=center_ky, scale=(ncol - 1) / self.mask_args['std_scale'])).astype(int)

                if (0 <= indx < nrow and 0 <= indy < ncol and temp_mask[t, indx, indy] == 1 and loss_mask[t, indx, indy] != 1):
                    loss_mask[t, indx, indy] = 1
                    count += 1

            train_mask[t] = input_mask[t] - loss_mask[t]
        return train_mask.astype(np.float32), loss_mask.astype(np.float32)


if __name__ == '__main__':
    import yaml
    from scipy.io import loadmat
    from utils import complex2real
    from visualize import visualize
    
    with open('config.yaml', 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)['dataset']
    del args['data_path'] 
    del args['csv_path'] 
    del args['supervision'] 

    image = loadmat('../data/images/fs_0030_3T_combined_0.mat')['xn']
    undersampler = Undersampler(fold='train', supervision='self', undersampling_args=args)
    data = undersampler(image)
    data['full'] = complex2real(image)
    data['train_image'] = (data['train_image'][0] + (1j * data['train_image'][1])).transpose(2, 0, 1)
    data['full'] = (data['full'][0] + (1j * data['full'][1])).transpose(2, 0, 1)

    images = np.concatenate((data['train_image'], data['full']), axis=-1)
    visualize(np.abs(images))

    '''
    data = np.load('masks.npy')
    mask = undersampler.cartesian_mask(image.shape)
    u_i, u_k =undersampler.undersample(image, mask)

    cx = int(undersampler.find_center_ind(u_k, axis=(0,1)))
    '''
