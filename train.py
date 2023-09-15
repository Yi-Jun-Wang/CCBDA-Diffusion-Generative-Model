import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from torchsummary import summary
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import MyDDPM, MyUNet
from dataset import TrainDataset, TransformSubset
from generate_image import save_DP_Image

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

dataset = TrainDataset('./datas')
indices = torch.randperm(
    len(dataset), generator=torch.Generator().manual_seed(42))

train_transform = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.),
    transforms.Lambda(lambda t: (t * 2) - 1),
])

train_dataset = TransformSubset(dataset, indices, train_transform)

train_loader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        drop_last=True)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

img_size = (1, 28, 28)
Time_step = 1000
Time_dim = 100

device = torch.device('cuda:0')
model = MyDDPM(MyUNet(Time_step, Time_dim), Time_step, device=device)
#model.apply(weight_init)
optimizer = Adam(model.parameters(), args.lr)

summary(MyUNet(Time_step, Time_dim).to(device), [img_size, (1,)])

total_epoch = 1
# checkpoint = torch.load('hw3.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# total_epoch = checkpoint['epoch']
# optimizer.param_groups[0]['lr'] = 1e-3
mse = nn.MSELoss()
for epoch in range(total_epoch, args.epochs + total_epoch):
    total_loss = 0
    with tqdm(train_loader, ncols=0, leave=False) as pbar:
        pbar.set_description(f"Epoch {epoch:3d}/{args.epochs:3d}")
        for Img in pbar:
            Img = Img.to(device)
            t = torch.randint(0, Time_step, (args.batch_size,), device=device)
            noise = torch.randn_like(Img).to(device)
            
            noisy_img = model.forward(Img, t, noise)
            noise_pred = model.backward(noisy_img, t)

            loss = mse(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix_str(f"loss: {loss.item():.4f}")
    
    avg_loss = total_loss/len(train_loader)

    print(", ".join([
        f"Epoch {epoch:3d}/{args.epochs:3d}",
        f"train_loss: {avg_loss:.4f}",
    ]))

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': avg_loss,},
        './hw3.pth')
    
    # if epoch % 20 == 0:
    #     save_DP_Image(f'sample_imgs/epoch_{epoch}_DP.png')