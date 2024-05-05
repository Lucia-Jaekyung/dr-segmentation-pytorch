import os
import argparse
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor, Compose, Resize, Normalize

from network.deeplabv3plus import DeepLabv3Plus
from network.ssmd_deeplabv3plus import SSMD_DeepLabv3Plus
from make_dataset import CustomImageDataset


def get_dataset(args):
    train_dir = args.train_dir
    val_dir = args.val_dir

    train_img_dir = os.path.join(train_dir, 'Original_Images')
    trian_recon_dir = train_img_dir
    train_micro_dir = os.path.join(train_dir, 'Microaneurysms')
    train_hemo_dir = os.path.join(train_dir, 'Hemohedges')
    train_hard_dir = os.path.join(train_dir, 'Hard_Exudates')
    train_soft_dir = os.path.join(train_dir, 'Soft_Exudates')

    val_img_dir = os.path.join(val_dir, 'Original_Images')
    val_recon_dir = val_img_dir
    val_micro_dir = os.path.join(val_dir, 'Microaneurysms')
    val_hemo_dir = os.path.join(val_dir, 'Hemohedges')
    val_hard_dir = os.path.join(val_dir, 'Hard_Exudates')
    val_soft_dir = os.path.join(val_dir, 'Soft_Exudates')

    train_dataset = CustomImageDataset(train_img_dir, trian_recon_dir, train_micro_dir, train_hemo_dir, train_hard_dir, train_soft_dir, transform=ToTensor())
    val_dataset = CustomImageDataset(val_img_dir, val_recon_dir, val_micro_dir, val_hemo_dir, val_hard_dir, val_soft_dir, transform=ToTensor())
    
    return train_dataset, val_dataset


def get_dataloader(train_dataset, val_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader


def trainer(args):
    train_dataset, val_dataset = get_dataset(args)
    train_dataloder, val_dataloader = get_dataloader(train_dataset, val_dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'deeplabv3plus':
        model = DeepLabv3Plus()
    elif args.model == 'ssmd_deeplabv3plus':
        model = SSMD_DeepLabv3Plus(in_planes=3, num_classes=2, os=16)
        # model.encoder.load_state_dict(torch.load('./models/encoder.pth'))
    else:
        raise ValueError('Invalid model name')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    recon_criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    alpha = 0.1
    beta = 0.8
    mask_list = ['micro', 'hemo', 'hard', 'soft']
    num_epochs = 150

    print('Start training...')
    for i in range(len(mask_list)):
        early_stopping_epochs = 7
        best_loss = float('inf')
        early_stopping_counter = 0
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for images, recon, micro, hemo, hard, soft in tqdm(train_dataloder):
                images, recon, micro, hemo, hard, soft = images.to(device).float(), recon.to(device).float(), micro.to(device).long(), hemo.to(device).long(), hard.to(device).long(), soft.to(device).long()
                x, x1, x2, x3, x4 = model(images)
                
                x_loss = recon_criterion(x, recon)
                x1_loss = criterion(x1, micro)
                x2_loss = criterion(x2, hemo)
                x3_loss = criterion(x3, hard)
                x4_loss = criterion(x4, soft)

                loss_list = [x_loss, x1_loss, x2_loss, x3_loss, x4_loss]

                seg_loss = beta * loss_list[i] + (1-beta) * sum(loss_list.pop(i))
                ssmd_loss = alpha * x_loss + (1-alpha) * seg_loss

                optimizer.zero_grad()
                ssmd_loss.backward()
                optimizer.step()
                train_loss += ssmd_loss.item()

            model.eval()
            val_loss = 0.0
            with torch.nn.grad():
                for images, recon, micro, hemo, soft in tqdm(val_dataloader):
                    images, recon, micro, hemo, hard, soft = images.to(device).float(), recon.to(device).float(), micro.to(device).long(), hemo.to(device).long(), hard.to(device).long(), soft.to(device).long()
                    x, x1, x2, x3, x4 = model(images)
                    
                    x_loss = recon_criterion(x, recon)
                    x1_loss = criterion(x1, micro)
                    x2_loss = criterion(x2, hemo)
                    x3_loss = criterion(x3, hard)
                    x4_loss = criterion(x4, soft)

                    loss_list = [x_loss, x1_loss, x2_loss, x3_loss, x4_loss]

                    seg_loss = beta * loss_list[i] + (1-beta) * sum(loss_list.pop(i))
                    ssmd_loss = alpha * x_loss + (1-alpha) * seg_loss

                    val_loss += ssmd_loss.item()

            if val_loss > best_loss:
                early_stopping_counter += 1
            else:
                best_loss = val_loss
                early_stopping_counter = 0
                torch.save(model.state_dict(), f'./models/ssmd_deeplabv3plus_{mask_list[i]}.pth')

            if early_stopping_counter >= early_stopping_epochs:
                print(f'Early stopping at epoch {epoch} for mask {mask_list[i]}')
                break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, help='path to the train dataset')
    parser.add_argument('--val_dir', type=str, help='path to the validation dataset')
    parser.add_argument('--model', type=str, default='ssmd_deeplabv3plus', help='model to train')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    trainer(args)