import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, recon_dir, micro_dir, hemo_dir, hard_dir, soft_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.recon_dir = recon_dir
        self.micro_dir = micro_dir
        self.hemo_dir = hemo_dir
        self.hard_dir = hard_dir
        self.soft_dir = soft_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_list = sorted(os.listdir(self.img_dir))
        image = cv2.imread(os.path.join(self.img_dir, img_list[idx]))
        image = self.preprocessing(image)
        image = image / 255

        recon_list = sorted(os.listdir(self.recon_dir))
        recon = cv2.imread(os.path.join(self.recon_dir, recon_list[idx]))
        recon = cv2.cvtColor(recon, cv2.COLOR_BGR2RGB)
        recon = cv2.resize(recon, (512, 512))
        recon = recon / 255

        micro_list = sorted(os.listdir(self.micro_dir))
        micro = cv2.imread(os.path.join(self.micro_dir, micro_list[idx]), cv2.IMREAD_GRAYSCALE)
        # micro = cv2.resize(micro, (512, 512))

        hemo_list = sorted(os.listdir(self.hemo_dir))
        hemo = cv2.imread(os.path.join(self.hemo_dir, hemo_list[idx]), cv2.IMREAD_GRAYSCALE)
        # hemo = cv2.resize(hemo, (512, 512))

        hard_list = sorted(os.listdir(self.hard_dir))
        hard = cv2.imread(os.path.join(self.hard_dir, hard_list[idx]), cv2.IMREAD_GRAYSCALE)
        # hard = cv2.resize(hard, (512, 512))

        soft_list = sorted(os.listdir(self.soft_dir))
        soft = cv2.imread(os.path.join(self.soft_dir, soft_list[idx]), cv2.IMREAD_GRAYSCALE)
        # soft = cv2.resize(soft, (512, 512))

        if self.transform:
            image = self.transform(image)
            recon = self.transform(recon)

        if self.target_transform:
            micro = self.transform(micro)
            hemo = self.transform(hemo)
            hard = self.transform(hard)
            soft = self.transform(soft)
        return image, recon, micro, hemo, hard, soft

    def preprocessing(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image[:, :, 0] = clahe.apply(image[:, :, 0])
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        image = cv2.resize(image, (512, 512))
        return image
    

def get_dataloader(data_dir, micro_dir, hemo_dir, hard_dir, soft_dir, batch_size=4):
    transform = ToTensor()
    dataset = CustomImageDataset(data_dir, micro_dir, hemo_dir, hard_dir, soft_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader