from torchvision.utils import save_image
from model import MyDDPM, MyUNet
from tqdm import tqdm
import os
import torch

@torch.no_grad()
def save_DP_Image(file_path, n_steps=1000, time_dim=100, device="cpu"):
    ddpm = MyDDPM(MyUNet(n_steps, time_dim), n_steps=n_steps, device=device)
    checkpoint = torch.load('hw3.pth')
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    img_size = (1, 28, 28)
    # Starting from random noise
    n_samples = 8
    record_step = n_steps//(n_samples-1)
    x = torch.zeros((n_samples, *img_size)).to(device)
    # x = torch.randn(n_samples, *img_size).to(device)
    out_img = (x + 1)/2
    step = 1
    for t in range(ddpm.n_steps)[::-1]:
        # Estimating noise to be removed
        time_tensor = (torch.ones(n_samples, 1, dtype=torch.long, device=device) * t)
        eta_theta = ddpm.backward(x, time_tensor)

        alpha_t = ddpm.alphas[t]
        alpha_t_bar = ddpm.alpha_bars[t]

        # Partially denoising the image
        x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

        if t > 0:
            z = torch.randn(n_samples, *img_size).to(device)

            beta_t = ddpm.betas[t]
            sigma_t = beta_t.sqrt()

            x = x + sigma_t * z

        if step % record_step == 0:
            normalized = x.clone()
            for i in range(len(normalized)):
                normalized[i] -= torch.min(normalized[i])
                normalized[i] *= 1. / torch.max(normalized[i])
            if out_img == None:
                out_img = normalized
            else:
                out_img = torch.cat((out_img, normalized), dim=0)
        step += 1

    save_image(out_img, file_path)

@torch.no_grad()
def save_10000_imgs(n_steps=1000, time_dim=100, device="cuda:0"):
    ddpm = MyDDPM(MyUNet(n_steps, time_dim), n_steps=n_steps, device=device)
    checkpoint = torch.load('hw3.pth')
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    img_size = (1, 28, 28)
    total_generated = 10000
    n_samples = 100
    step = 1
    for i in tqdm(range(total_generated//n_samples)):
        x = torch.zeros((n_samples, *img_size)).to(device)
        # x = torch.randn(n_samples, *img_size).to(device)
        for t in range(ddpm.n_steps)[::-1]:
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, *img_size).to(device)

                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                x = x + sigma_t * z

            if t == 0:
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 1. / torch.max(normalized[i])
                    save_image(normalized[i], f"gene_images/{step:05d}.png")
                    step += 1

if __name__ == '__main__':
    device = torch.device('cuda:0')
    if not os.path.isdir("./gene_images"):
        os.mkdir("./gene_images")
    save_DP_Image(f'./311511044.png', device=device)
    save_10000_imgs(n_steps=1000, time_dim=100, device=device)