import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import dataset, transforms, models
from torchvision.transforms import ToTensor

from network.ssmd_deeplabv3plus import SSMDDeepLabv3plus
from make_dataset import CustomImageDataset


def dice_score(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))


def get_dataset(args):
    test_dir = args.test_dir

    test_img_dir = os.path.join(test_dir, 'Original_Images')
    test_recon_dir = test_img_dir
    test_micro_dir = os.path.join(test_dir, 'Microaneurysms')
    test_hemo_dir = os.path.join(test_dir, 'Hemohedges')
    test_hard_dir = os.path.join(test_dir, 'Hard_Exudates')
    test_soft_dir = os.path.join(test_dir, 'Soft_Exudates')

    test_dataset = CustomImageDataset(test_img_dir, test_recon_dir, test_micro_dir, test_hemo_dir, test_hard_dir, test_soft_dir, transform=ToTensor())

    return test_dataset


def get_dataloader(test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_dataloader


def tester(args):
    test_dataset = get_dataset(args)
    test_dataloader = get_dataloader(test_dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SSMDDeepLabv3plus(in_planes=3, n_classes=2, os=16)
    model.load_state_dict(torch.load('./models/ssmd_deeplabv3plus.pth'))
    model.eval()
    with torch.no_grad():
        micro_score_list = []
        hemo_score_list = []
        hard_score_list = []
        soft_score_list = []

        for image, recon, micro, hemo, hard, soft in tqdm(test_dataloader):
            image = image.float()
            # mask = mask.float()
            recon_pred, micro_pred, hemo_pred, hard_pred, soft_pred = model(image)
            _, micro_pred = torch.max(micro_pred, 1)
            _, hemo_pred = torch.max(hemo_pred, 1)
            _, hard_pred = torch.max(hard_pred, 1)
            _, soft_pred = torch.max(soft_pred, 1)

            # mask = torch.cat((micro_pred, hemo_pred, hard_pred, soft_pred), dim=1)

            micro_score = dice_score(micro.squeeze().cpu().numpy(), micro_pred.squeeze().cpu().numpy())
            hemo_score = dice_score(hemo.squeeze().cpu().numpy(), hemo_pred.squeeze().cpu().numpy())
            hard_score = dice_score(hard.squeeze().cpu().numpy(), hard_pred.squeeze().cpu().numpy())
            soft_score = dice_score(soft.squeeze().cpu().numpy(), soft_pred.squeeze().cpu().numpy())

            if micro_score > 0:
                micro_score_list.append(micro_score)
            if hemo_score > 0:
                hemo_score_list.append(hemo_score)
            if hard_score > 0:
                hard_score_list.append(hard_score)
            if soft_score > 0:
                soft_score_list.append(soft_score)

        print(f'Micro Dice Score (Mean): {np.mean(micro_score_list)}, Micro Dice Score (Max): {np.max(micro_score_list)}')
        print(f'Hemo Dice Score (Mean): {np.mean(hemo_score_list)}, Hemo Dice Score (Max): {np.max(hemo_score_list)}')
        print(f'Hard Dice Score (Mean): {np.mean(hard_score_list)}, Hard Dice Score (Max): {np.max(hard_score_list)}')
        print(f'Soft Dice Score (Mean): {np.mean(soft_score_list)}, Soft Dice Score (Max): {np.max(soft_score_list)}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, help='path to the test dataset')
    parser.add_argument('--model', type=str, help='model name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    tester(args)