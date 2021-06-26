import os
import math
import numpy as np
from tqdm import tqdm
from utils import resize, fft2c
from scipy.io import loadmat, savemat
from read_ocmr import read_ocmr as read
from ismrmrdtools import coils, transform

def preprocess(data_path, out_path, csm_path=None, ref_size=None):
    pbar = tqdm(os.listdir(data_path))
    for f in pbar:
        fname = f.split('.')
        if fname[-1] != 'h5':
            continue

        # Read k-space data in shape {'kx ky kz coil phase set slice rep avg'}
        kData, _ = read(os.path.join(data_path, f))
        kData = np.mean(kData, axis=-1)
        kData = kData[..., 0] # Remove Rep dimension

        im_coils = transform.transform_kspace_to_image(kData, [0,1]) # IFFT (2D image)
        
        # sample shape: {nx ny nz coil phase set slice}
        RO = im_coils.shape[0]
        im_coils = im_coils[math.floor(RO/4):math.floor(RO/4*3)] # Remove RO oversampling

        num_coils = im_coils.shape[3]
        num_slices = im_coils.shape[-1]
        num_time_steps = im_coils.shape[4]

        # Estimate the coil sensitivity maps using the center slice and time step 
        sample_frame = np.squeeze(im_coils[:, :, :, :, num_time_steps // 2, :, num_slices // 2]).transpose(2, 0, 1)

        csm_est, _ = coils.calculate_csm_inati_iter(sample_frame)
        if csm_path != None:
            if ref_size != None:
                tobesaved = resize(csm_est, ref_size)
                savemat(os.path.join(csm_path, fname[0] + '_csm.mat'), {'csm': tobesaved})
            else:
                savemat(os.path.join(csm_path, fname[0] + '_csm.mat'), {'csm': csm_est})

        if ref_size != None:
            output_shape = (num_time_steps, num_coils, ref_size[0], ref_size[1]) 
        else:
            output_shape = (num_time_steps, num_coils, im_coils.shape[0], im_coils.shape[1]) 

        for i in range(num_slices):
            pbar.set_description(fname[0] + ' [' + str(i + 1) + '/' + str(num_slices) + ']')
            im_slice = np.squeeze(im_coils[..., i])
            output = np.empty(output_shape, dtype=im_coils.dtype)
            for t in range(num_time_steps):
                frame = im_slice[..., t].transpose(2, 0, 1)
                
                # frame = (frame * np.conj(csm_est)).sum(0)
                # Resize the frame if a reference size is provided
                if ref_size != None:
                    temp_frame = np.empty((num_coils, ref_size[0], ref_size[1]), dtype=frame.dtype)
                    for c in range(num_coils):
                        temp_frame[c] = resize(frame[c], ref_size)
                    frame = temp_frame
                output[t] = frame

            # Normalize the k-space values in range [0, 1]
            k_space = fft2c(output)
            k_space = k_space / np.max(np.abs(k_space)[:])
            savemat(os.path.join(out_path, fname[0] + '_k_' + str(i) + '.mat'), {'xn': k_space})
            # savemat(os.path.join(out_path, fname[0] + '_combined_' + str(i) + '.mat'), {'xn': output})


if __name__ == "__main__":
    from visualize import visualize

    data_path = '../../ocmr/data/fullysampled'
    out_path = '../../ocmr/data/k_space'
    csm_path = None # '../../ocmr/data/csm'
    ref_size = (144, 144)

    preprocess(data_path, out_path, csm_path, ref_size)
