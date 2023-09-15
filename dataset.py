import glob
import os
from torchvision.io import read_image, ImageReadMode
import torch
from torch.utils.data import Dataset, Subset

class TrainDataset(Dataset):
    def __init__(self, root='./datas'):
        self.paths = sorted(
            glob.glob(os.path.join(root, "*.png"), recursive=True))  

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = read_image(self.paths[idx], ImageReadMode.GRAY).to(torch.float)
        return img

class TransformSubset(Subset):  # Subset 使用indices的取樣順序從dataset取出data.
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img = super().__getitem__(idx)  # dataset[indices[idx]]
        img = self.transform(img)
        return img

if __name__ == '__main__':
    img = read_image('datas/00001.png')/255*2 - 1
    print(torch.min(img), torch.max(img))