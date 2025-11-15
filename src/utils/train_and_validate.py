import torch
import torch.nn as nn
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from .losses import psnr



def train_one_epoch(model, loader, optimizer, criterion, device, epoch=0, log_interval=50, supress_tqdm=True):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train Epoch {epoch}", disable=supress_tqdm)
    for batch_idx, (noisy, clean) in pbar:
        noisy, clean = noisy.to(device), clean.to(device)

        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()
        mse = nn.functional.mse_loss(output, clean).item()
        running_psnr += psnr(mse)

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            avg_psnr = running_psnr / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "psnr": f"{avg_psnr:.2f}dB"})

    avg_loss = running_loss / len(loader)
    avg_psnr = running_psnr / len(loader)
    return avg_loss, avg_psnr


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, epoch=0, supress_tqdm=True):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Val Epoch {epoch}", disable=supress_tqdm)
    for batch_idx, (noisy, clean) in pbar:
        noisy, clean = noisy.to(device), clean.to(device)

        output = model(noisy)
        loss = criterion(output, clean)

        running_loss += loss.item()
        mse = nn.functional.mse_loss(output, clean).item()
        running_psnr += psnr(mse)

        avg_loss = running_loss / (batch_idx + 1)
        avg_psnr = running_psnr / (batch_idx + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "psnr": f"{avg_psnr:.2f}dB"})

    avg_loss = running_loss / len(loader)
    avg_psnr = running_psnr / len(loader)
    return avg_loss, avg_psnr


@torch.no_grad()
def visualize_one_batch(model, loader, device, fold, max_images=8, supress_tqdm=True):
    model.eval()
    n_images = 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Visualize", disable=supress_tqdm)
    for batch_idx, (noisy, clean) in pbar:
        noisy, clean = noisy.to(device), clean.to(device)

        output = model(noisy)
        noisy = noisy[:max_images].cpu()
        output = output[:max_images].cpu()
        clean = clean[:max_images].cpu()
        for idx in range(output.shape[0]):
            plt.subplots(2, 2, figsize=(10, 10))

            plt.subplot(2, 2, 1)
            plt.imshow(clean[idx].permute(1, 2, 0))

            plt.subplot(2, 2, 2)
            plt.imshow(output[idx].permute(1, 2, 0))

            plt.subplot(2, 2, 3)
            plt.imshow(noisy[idx].permute(1, 2, 0))

            plt.subplot(2, 2, 4)
            plt.imshow(output[idx].permute(1, 2, 0)-clean[idx].permute(1, 2, 0)+0.5)
            
            n_images+=1
            plt.savefig(f'/kaggle/working/{fold}_{batch_idx}_{idx}.jpg')
        if n_images >= max_images:
            return True