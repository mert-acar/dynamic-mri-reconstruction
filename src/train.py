import os
import yaml
import torch
import models
from time import time
from tqdm import tqdm
from tabulate import tabulate
from dataset import OCMRDataset
from shutil import rmtree, copyfile
from utils import complex_psnr, ssim_score, real2complex, time_trim, shrink


def checkpoint(model, optimizer, path):
    print('SAVING MODEL...')
    dict2save = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(dict2save, os.path.join(path, 'checkpoint'))
    

def test(model, dataloader, model_args, cuda, supervision='full'):
    total_loss = 0
    base_psnr = 0
    test_psnr = 0
    ssim = 0
    model.eval()
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for batch in tqdm(dataloader, "TESTING", total=len(dataloader)):
            bsize = batch['train_image'].shape[0]
            
            if cuda:
                for key in batch:
                    batch[key] = batch[key].cuda()
            
            if model_args['model_type'] == 'cascade':
                batch = shrink(batch)
                inp = batch['train_image'].clone()
                output = model(**batch)
            elif model_args['model_type'] == 'drn':
                inp = batch['train_image'].clone()
                output = model(**batch)
            elif model_args['model_type'] == 'modrn':
                batch = time_trim(batch, model_args['T'])
                if batch['train_image'].shape[-1] == 0:
                    continue
                inp = batch['train_image'].clone()
                idx = list(range(-1, inp.shape[-1], model_args['T']))
                idx[0] = 0
                if supervision == 'self':
                    ref = batch['train_image'][..., idx]
                else:
                    ref = batch['full'][..., idx]

                ref = ref.float()
                if train_args['cuda']:
                    ref = ref.cuda()
                output = model(ref=ref, **batch)

            if supervision == 'self':
                loss = criterion(output[1], batch['loss_k'])
                output = output[0]
            else:
                loss = criterion(output, batch['full'])

            total_loss += loss.item() * bsize

            for im_i, und_i, pred_i in zip(real2complex(batch['full'].cpu().detach().numpy()),
                                           real2complex(inp.cpu().detach().numpy()), 
                                           real2complex(output.cpu().detach().numpy())):
                base_psnr += complex_psnr(im_i, und_i)
                test_psnr += complex_psnr(im_i, pred_i)
                ssim += ssim_score(im_i, pred_i)

    total_loss = total_loss / len(dataloader.dataset)
    base_psnr /= len(dataloader.dataset)
    test_psnr /= len(dataloader.dataset)
    ssim /= len(dataloader.dataset)
    print('Loss:    \t{:.4f} x 1e-3'.format(total_loss * 1000))
    print('Base PSNR:\t{:.4f}'.format(base_psnr))
    print('Test PSNR:\t{:.4f}'.format(test_psnr))
    print('SSIM:      \t{:.4f}'.format(ssim))

    scores = {
        'Testing Loss': total_loss,
        'Base PSNR': base_psnr,
        'Test PSNR': test_psnr,
        'SSIM': ssim
    }
    return scores


def train_model(model, dataloaders, optimizer, criterion, lr_scheduler, model_args, train_args, supervision='full'):
    before = time()
    best_loss = 9999999
    best_epoch = -1
    loss_table = []
    
    for epoch in range(1, train_args['num_epochs'] + 1):
        print('-' * 20)
        print('Epoch {} / {}'.format(epoch, train_args['num_epochs']))
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_error = 0
            base_psnr = 0
            test_psnr = 0
            ssim = 0
            with torch.set_grad_enabled(phase == 'train'):
                for batch in tqdm(dataloaders[phase], phase + 'ing...', total=(len(dataloaders[phase]))):
                    bsize = batch['train_image'].shape[0]
                    optimizer.zero_grad()

                    if train_args['cuda']:
                        for key in batch:
                            batch[key] = batch[key].cuda()

                    # Trim the time series to an integer multiple of the motion period
                    if model_args['model_type'] == 'cascade':
                        batch = shrink(batch)
                        inp = batch['train_image'].clone()
                        output = model(**batch)
                    elif model_args['model_type'] == 'drn':
                        inp = batch['train_image'].clone()
                        output = model(**batch)
                    elif model_args['model_type'] == 'modrn':
                        batch = time_trim(batch, model_args['T'])
                        if batch['train_image'].shape[-1] == 0:
                            continue
                        inp = batch['train_image'].clone()
                        idx = list(range(-1, inp.shape[-1], model_args['T']))
                        idx[0] = 0
                        if supervision == 'self':
                            ref = batch['train_image'][..., idx]
                        else:
                            ref = batc['full'][..., idx]
                        ref = ref.float()
                        if train_args['cuda']:
                            ref = ref.cuda()
                        output = model(ref=ref, **batch)

                    if supervision == 'self':
                        loss = criterion(output[1], batch['loss_k'])
                        output = output[0]
                    else:
                        loss = criterion(output, batch['full'])

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        for im_i, und_i, pred_i in zip(real2complex(batch['full'].cpu().detach().numpy()),
                                                       real2complex(inp.cpu().detach().numpy()), 
                                                       real2complex(output.cpu().detach().numpy())):
                            base_psnr += complex_psnr(im_i, und_i)
                            test_psnr += complex_psnr(im_i, pred_i)
                            ssim += ssim_score(im_i, pred_i)
                    running_error += loss.item() * bsize
            running_error = running_error / len(dataloaders[phase].dataset)
            print('Loss:    \t{:.4f} x 1e-3'.format(running_error * 1000))
            if phase == 'test':
                base_psnr /= len(dataloaders[phase].dataset)
                test_psnr /= len(dataloaders[phase].dataset)
                ssim /= len(dataloaders[phase].dataset)
                print('Base PSNR:\t{:.4f}'.format(base_psnr))
                print('Test PSNR:\t{:.4f}'.format(test_psnr))
                print('SSIM:      \t{:.4f}'.format(ssim))
                loss_table.append([epoch, last_train_loss, running_error, base_psnr, test_psnr, ssim])
                lr_scheduler.step(running_error)
                if running_error < best_loss:
                    best_loss = running_error
                    best_epoch = epoch
                    checkpoint(model, optimizer, train_args['output_path'])
            else:
                last_train_loss = running_error

        if(train_args['early_stop'] != 0) and ((epoch - best_epoch) >= train_args['early_stop']):
            print('No improvements in', train_args['early_stop'], 'epochs, break...')
            break

    t = tabulate(loss_table, headers=['Epoch', 'Train Loss', 'Test Loss', 'Base PSNR', 'Test PSNR', 'SSIM'])
    with open(os.path.join(train_args['output_path'], 'log_train.txt'), 'w') as f:
        f.write(t)
    elapsed = time() - before
    print("Training complete in {:.0f}m {:.0f}s".format(elapsed // 60, elapsed % 60))


def loss_func(u, v):
    l2 = torch.linalg.norm(v - u)
    l2n = torch.linalg.norm(v)
    l2 = l2 / l2n
    
    l1 = torch.linalg.norm(torch.flatten(v - u), ord=1)
    l1n = torch.linalg.norm(torch.flatten(v), ord=1)
    l1 = l1 / l1n
    return (l1 + l2) * 0.5


def train(train_args, data_args, model_args):
    dataloaders = {
        'train':torch.utils.data.DataLoader(
            OCMRDataset(fold='train', **data_args),
            batch_size=data_args['batch_size'],
            shuffle=True,
        ),
        'test':torch.utils.data.DataLoader(
            OCMRDataset(fold='test', **data_args),
            batch_size=data_args['batch_size'],
            shuffle=False,
        )
    }

    if model_type == 'cascadenet':
        model = models.CascadeNetwork(**model_args)
    elif model_type == 'drn':
        model = models.DRN(**model_args['drn'])
    else:
        model = models.MODRN(**model_args)
    
    if train_args['cuda']:
        model.cuda()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_args['learning_rate'],
        weight_decay=train_args['l2']
    )

    criterion = loss_func
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=train_args['lr_scheduler_factor'],
        patience=train_args['lr_scheduler_patience'],
        threshold=train_args['lr_scheduler_threshold']
    )

    print('Training starting with', len(dataloaders['train'].dataset), 'training and', len(dataloaders['test'].dataset), 'testing data...')
    train_model(model, dataloaders, optimizer, criterion, lr_scheduler, model_args, train_args, data_args['supervision'])

    check = torch.load(os.path.join(train_args['output_path'], 'checkpoint'))['model']
    model.load_state_dict(check)
    scores = test(model, dataloaders['test'], model_args, train_args['cuda'], data_args['supervision'])

    t = [list(item) for item in scores.items()]
    t = tabulate(t, headers=['','Score'])
    with open(os.path.join(train_args['output_path'], 'scores.txt'), 'w') as f:
        f.write(t)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model_type', type=str, default='drn', required=False)
    parser.add_argument('-c', '--config_path', type=str, default='models/configurations/motion_guided.yaml', required=False)
    model_config = parser.parse_args()
    
    model_type = model_config.model_type
    with open(model_config.config_path, 'r') as f:
        model_args = yaml.load(f, Loader=yaml.FullLoader)
    model_args['model_type'] = model_type

    with open('config.yaml', 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader) 

    train_args = args['train']
    data_args = args['dataset']
    if not isinstance(train_args['cuda'], bool):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(train_args['cuda'])
        print("Using GPU device:", train_args['cuda'])
        train_args['cuda'] = True

    if os.path.exists(train_args['output_path']):
        c = input('Output path ' + train_args['output_path'] + ' is not empty! Do you want to delete the existing folder? [y/n]: ')
        if c.lower() == 'y':
            rmtree(train_args['output_path'])
        else:
            print('Exit!')
            raise SystemExit

    os.mkdir(train_args['output_path'])
    copyfile('config.yaml', os.path.join(train_args['output_path'], 'ExperimentSummary.yaml')) 
    copyfile(model_config.config_path, os.path.join(train_args['output_path'], 'ModelConfig.yaml')) 
    train(train_args, data_args, model_args)
