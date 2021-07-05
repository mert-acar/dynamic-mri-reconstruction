import numpy as np
from skimage.metrics import structural_similarity
from numpy.fft import ifftshift, fft2, ifft2, fftshift


def complex2real(x):
    '''
    Parameter
    ---------
    x: ndarray
        assumes at least 2d. Last 2D axes are split in terms of real and imag
        3d complex valued tensor (nt, nx, ny)
    Returns
    -------
    y: 4d tensor (2, nx, ny, nt)
    '''
    x_real = np.expand_dims(np.real(x), 0)
    x_imag = np.expand_dims(np.imag(x), 0)
    y = np.concatenate((x_real, x_imag), 0).astype(np.float32)
    return y.transpose(0, 2, 3, 1)


def real2complex(x):
    '''
        Converts from array of the form (n, 2, nx, ny, nt) to complex valued (n, nx, ny, nt)
    '''
    x = x[:, 0] + (1j * x[:, 1])
    return x


def complex_psnr(x, y, peak='max'):
    '''
    x: reference image
    y: reconstructed image
    peak: normalised or max
    Notice that ``abs'' squares
    '''
    mse = np.mean(np.abs(x - y)**2)
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(x))**2/(mse + 1e-8))
    else:
        return 10*np.log10(1./(mse + 1e-8))


def ssim_score(x, y):
    score = 0
    tx = np.abs(x) / np.max(np.abs(x))
    ty = np.abs(y) / np.max(np.abs(y))
    for i in range(x.shape[-1]):
        score += structural_similarity(tx[..., i], ty[..., i])
    score = score / x.shape[-1]
    return score


def time_trim(batch, T):
    res = batch['full'].shape[-1] % T
    for key in batch:
        if len(batch[key].shape) == 4:
            batch[key] = batch[key][:, :-res]
        else:
            batch[key] = batch[key][..., :-res]
    return batch


def resize(data, ref_size):
    import cv2
    if data.ndim == 2:
        data = np.expand_dims(data, 0)
        frame2d = True
    else:
        frame2d = False

    temp_frame = np.empty((data.shape[0], ref_size[0], ref_size[1]), dtype=data.dtype)
    for c in range(data.shape[0]):
        frame = data[c]
        frame_r = np.real(frame)
        frame_r = cv2.resize(frame_r, ref_size)
        frame_i = np.imag(frame)
        frame_i = cv2.resize(frame_i, ref_size)
        frame = frame_r + (1j * frame_i)
        temp_frame[c] = frame
    data = temp_frame

    if frame2d:
        data = data[0]
    return data


def fft2c(x, norm='ortho', axes=(-2, -1)):
    return fftshift(fft2(ifftshift(x, axes=axes), norm=norm), axes=axes)


def ifft2c(x, norm='ortho', axes=(-2, -1)):
    return fftshift(ifft2(ifftshift(x, axes=axes), norm=norm), axes=axes)


def shrink(data):
    for key in data:
        if data[key].ndim == 5:
            data[key] = data[key][..., 0]
        else:
            data[key] = data[key][:, 0]
    return data


def create_csv(data_dir, output_dir, train_split=0.85):
    import os
    import numpy as np
    import pandas as pd
    data = {'Filename': [], 'Fold': []}
    filelist = [f for f in os.listdir(data_dir) if f.split('.')[-1] == 'mat']
    data['Filename'] = np.random.permutation(filelist)
    data['Fold'] = ['train'] * int(len(filelist) * train_split)
    data['Fold'] += ['test'] * (len(filelist) - len(data['Fold']))
    data = pd.DataFrame(data)
    data = data.set_index('Filename')
    data.to_csv(output_dir)


if __name__ == '__main__':
    # CSV Creation
    create_csv('../data/', '../metadata/train-test-images.csv')
