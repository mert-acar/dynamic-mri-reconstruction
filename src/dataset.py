import os
import pandas as pd
from scipy.io import loadmat
from utils import complex2real
from torch.utils.data import Dataset
from undersampler import Undersampler


class OCMRDataset(Dataset):
    def __init__(self, data_path, csv_path, supervision='full', fold=None, **mask_args):
        self.data_path = data_path
        self.df = pd.read_csv(csv_path)
        if fold:
            self.df = self.df[self.df['Fold'] == fold]
        self.df = self.df.set_index('Filename')
        self.undersampler = Undersampler(fold, supervision, mask_args)
    

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        filename = self.df.index[idx]
        image = loadmat(os.path.join(self.data_path, filename))['xn']
        data = self.undersampler(image) 
        data['full'] = complex2real(image)
        return data


if __name__ == '__main__':
    import yaml
    import numpy as np
    from utils import real2complex
    from visualize import visualize
    with open('config.yaml', 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    
    dataset = OCMRDataset(fold='train', **args['dataset'])
    data = dataset[0]
    for key in data:
        print(key + ' shape:' , data[key].shape, data[key].dtype)
    
    image = real2complex(data['train_image']).transpose(2, 0, 1)
    full = real2complex(data['full']).transpose(2, 0, 1)
    
    images = np.concatenate((image, full), axis=-1)
    visualize(np.abs(images), fps=24)
    masks = np.concatenate((data['train_mask'], data['loss_mask']), axis=-1)
    visualize(masks, fps=24)
