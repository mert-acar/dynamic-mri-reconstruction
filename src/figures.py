import os
import yaml
import torch 
import numpy as np
from helpers import *
from prepare import *
from models import DRN, MODRN
from dataset import OCMRDataset
from scipy.io import loadmat, savemat
from cascadenet import CascadeNetwork

np.random.seed(9001)

def fetch_data_cascade(args):
    data_path = '../../ocmr/data/images/fs_0066_1_5T_combined_0.mat'
    image = loadmat(data_path)['xn']
    image = image[image.shape[0] // 2]
    mask_shape = [1, image.shape[0], image.shape[1]]
    mask = cartesian_mask(mask_shape, args['acceleration_factor'], args['sample_n'], args['centred']).squeeze()
    u_image, u_k = undersample(image, mask)
    
    image_r = np.expand_dims(np.real(image), 0)
    image_i = np.expand_dims(np.imag(image), 0)
    image = np.concatenate((image_r, image_i), 0).astype(np.float32)

    u_image_r = np.expand_dims(np.real(u_image), 0)
    u_image_i = np.expand_dims(np.imag(u_image), 0)
    u_image = np.concatenate((u_image_r, u_image_i), 0).astype(np.float32)

    return {
        'image': u_image,
        'k': u_k,
        'mask': mask.astype(np.float32),
        'full': image
    }

def fetch_data(args):
    data_path = '../../ocmr/data/images/fs_0066_1_5T_combined_0.mat'
    image = loadmat(data_path)['xn']
    mask = cartesian_mask(image.shape, args['acceleration_factor'], args['sample_n'], args['centred'])
    u_image, u_k = undersample(image, mask)
    return {
        'image': complex2real(u_image),
        'k': u_k,
        'mask': mask.astype(np.float32),
        'full': complex2real(image)
    }


if __name__ == '__main__':
    model_path = '../logs/unsupervised_drn_x8'
    desired_name = 'crnn_x4'
    with open(os.path.join(model_path, 'ExperimentSummary.yaml'), 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    data = fetch_data(args['dataset'])
    np.save('unsx8.npz', data['image'][..., data['image'].shape[-1] // 2])
    raise SystemExit 

    criterion = torch.nn.MSELoss()
    model_type = model_path.split('/')[-1]
    if 'modrn' in model_type:
        data = fetch_data(args['dataset'])
        frame_idx = data['full'].shape[-1] // 2
        for key in data:
            data[key] = torch.unsqueeze(torch.from_numpy(data[key]), 0).cuda()
        data = time_trim(data, args['network']['T'])

        idx = list(range(-1, data['image'].shape[-1], args['network']['T']))
        idx[0] = 0
        if 'under' in model_type:
            ref = data['image'][..., idx].float()
        else:
            ref = data['full'][..., idx].float()

        model = MODRN(**args['network'])
        state = torch.load(os.path.join(model_path, 'checkpoint'))['model']
        model.load_state_dict(state)
        model.eval()
        model.cuda()

        output = model(data['image'], data['k'], data['mask'], ref)

    elif 'drn' in model_type:
        data = fetch_data(args['dataset'])
        frame_idx = data['full'].shape[-1] // 2
        for key in data:
            data[key] = torch.unsqueeze(torch.from_numpy(data[key]), 0).cuda()
        model = DRN(**args['network']['drn'])
        state = torch.load(os.path.join(model_path, 'checkpoint'))['model']
        model.load_state_dict(state)
        model.eval()
        model.cuda()
        output = model(data['image'], data['k'], data['mask'])

    elif 'cascade' in model_type:
        data = fetch_data_cascade(args['dataset'])
        for key in data:
            data[key] = torch.unsqueeze(torch.from_numpy(data[key]), 0).cuda()
        model = CascadeNetwork(**args['network'])
        state = torch.load(os.path.join(model_path, 'checkpoint'))['model']
        model.load_state_dict(state)
        model.eval()
        model.cuda()
        output = model(data['image'], data['k'], data['mask'])

    loss = np.around(criterion(output, data['full']).item() * 1e3, 5)
    im_ref = real2complex(data['full'].cpu().detach().numpy())[0]
    im_pred = real2complex(output.cpu().detach().numpy())[0]
    psnr = np.around(complex_psnr(im_ref, im_pred, 'max'), 5)
    ssim_score = np.around(ssim(im_ref, im_pred), 5)

    print(desired_name)
    print('-' * 10)
    print("Loss (x 1e3):\t", loss)
    print("PSNR:\t\t", psnr)
    print("SSIM:\t\t", ssim_score)
    
    if 'cascade' in model_type:
        im_ref = np.abs(im_ref)
        im_pred = np.abs(im_pred)
    else:
        im_ref = np.abs(im_ref[..., frame_idx])
        im_pred = np.abs(im_pred[..., frame_idx])

    error = np.abs((im_ref - im_pred) ** 2)

    figures = {
        'ref': im_ref,
        'pred': im_pred,
        'error': error
    }
    target_dir = os.path.join('../figures', '_'.join(list(map(str, [desired_name, loss, psnr, ssim_score]))) + '.mat')
    savemat(target_dir, figures)
