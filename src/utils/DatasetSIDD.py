
import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F 
import numpy as np
from tqdm import tqdm 
from sklearn.model_selection import KFold
import pandas as pd

class DatasetSIDD(Dataset):
    def __init__(self, scene_folders, patch_size=256, crop_size=2560,
                validation=False, transform=None, max_images=0, supress_tqdm=True):
        """
        scene_folders: list of scene directories
        """
        self.samples = []
        self.patch_size = patch_size
        self.validation = validation
        self.transform = transform

        for scene in scene_folders:
            noisy_imgs = sorted(glob.glob(os.path.join(scene, "*NOISY_SRGB_*.PNG")))
            gt_imgs    = sorted(glob.glob(os.path.join(scene, "*GT_SRGB_*.PNG")))

            for noisy_path, gt_path in zip(noisy_imgs, gt_imgs):
                self.samples.append((noisy_path, gt_path))

        # --- Pre-loading images ---
        self.noisy_img = []
        self.gt_img = []
        self.image_name = [] 
        self.supress_tqdm = supress_tqdm
        print(f"Loading {'validation' if validation else 'training'} data...")
        for noisy_path, gt_path in tqdm(self.samples, dissable=supress_tqdm):
            noisy = Image.open(noisy_path).convert("RGB")
            
            width, height = noisy.size  
            left = (width - crop_size)/2
            top = (height - crop_size)/2
            right = (width + crop_size)/2
            bottom = (height + crop_size)/2

            noisy = noisy.crop((left, top, right, bottom))
            noisy = np.asarray(noisy).copy()

            clean = Image.open(gt_path).convert("RGB")
            clean = clean.crop((left, top, right, bottom))
            clean = np.asarray(clean).copy()
            self.noisy_img.append(noisy)
            self.gt_img.append(clean)

            self.image_name.append((noisy_path, gt_path))
            if max_images and len(self.noisy_img) >= max_images:
                break



    def __len__(self):
        return len(self.noisy_img)


    def __getitem__(self, idx):
        noisy = self.noisy_img[idx]
        clean = self.gt_img[idx]
    
        # Convert to tensor for cropping
        noisy = transforms.ToTensor()(noisy)
        clean = transforms.ToTensor()(clean)
    
        # --- Random or center crop (aligned) ---
        _, H, W = noisy.shape
        ps = self.patch_size
        
        # Default to top-left if image is smaller than patch size (shouldn't happen with SIDD)
        top, left = 0, 0
        
        if H >= ps and W >= ps:
            if self.validation:
                top = (H - ps) // 2
                left = (W - ps) // 2
            else:
                top = random.randint(0, H - ps)
                left = random.randint(0, W - ps)
                
        noisy = noisy[:, top:top+ps, left:left+ps]
        clean = clean[:, top:top+ps, left:left+ps]
    
        # --- Random flips and rotations ---
        if not self.validation:
            # Horizontal flip
            if random.random() < 0.5:
                noisy = F.hflip(noisy)
                clean = F.hflip(clean)
    
            # Vertical flip
            if random.random() < 0.5:
                noisy = F.vflip(noisy)
                clean = F.vflip(clean)
    
            # 0°, 90°, 180°, or 270° rotation
            k = random.randint(0, 3)
            if k:
                noisy = torch.rot90(noisy, k, [1, 2])
                clean = torch.rot90(clean, k, [1, 2])
    
        # --- Optional additional transform ---
        if self.transform:
            noisy = self.transform(noisy)
            clean = self.transform(clean)
    
        return noisy, clean
    

def get_k_fold_datasets(root_dir, k_folds=5, patch_size=128, seed=42,  max_images=0, supress_tqdm=True):
    """
    A generator that yields train and validation datasets for each k-fold split.
    Splits are made at the SCENE level.
    """
    # Get all scene folders
    all_scene_folders = sorted(glob.glob(os.path.join(root_dir, "*")))
    
    # Use numpy for easier indexing
    all_scenes_np = np.array(all_scene_folders)

    # Initialize the K-Fold splitter
    # shuffle=True ensures a random (but reproducible) split
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    print(f"Total scenes: {len(all_scenes_np)}. Using {k_folds}-Fold Cross-Validation.")

    # kf.split() yields indices for train and test (which we use as val)
    for fold, (train_indices, val_indices) in enumerate(kf.split(all_scenes_np)):
        
        print(f"\n--- FOLD {fold + 1}/{k_folds} ---")
        
        # Select the scene folders for this fold
        train_scenes = list(all_scenes_np[train_indices])
        val_scenes = list(all_scenes_np[val_indices])

        print(f"Train scenes: {len(train_scenes)}, Val scenes: {len(val_scenes)}")

        # Create the Dataset objects using your class
        train_dataset = DatasetSIDD(train_scenes, 
                                     patch_size=patch_size, 
                                     validation=False,
                                     max_images=max_images,
                                     supress_tqdm=supress_tqdm)
                                     
        val_dataset = DatasetSIDD(val_scenes, 
                                   patch_size=patch_size, 
                                   validation=True,
                                   max_images=max_images,
                                     supress_tqdm=supress_tqdm)
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # Yield the datasets for this fold
        yield train_dataset, val_dataset


def make_patches(image: torch.tensor, patch_size=256):
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, 256, 256)
    return patches

def validate_model(val_dataset, model, criterion, device):
    model = model.eval()
    loss_list = []
    for images, (noisy, gt) in zip(val_dataset.image_name, val_dataset):
        noisy_patches = make_patches(noisy)
        gt_patches = make_patches(gt)
        noisy_patches = noisy_patches.to(device)

        for patch_number, (noisy_patch, gt_patch) in enumerate(zip(noisy_patches, gt_patches)):
            noisy_patch = noisy_patch.unsqueeze(0)
            gt_patch = gt_patch.unsqueeze(0)
            with torch.no_grad():
                pred = model(noisy_patch)
            loss = criterion(pred, gt_patch).item()
            gtloss = criterion(noisy_patch, gt_patch).item()
            loss_list.append({"images": images, "patch_number": patch_number, "loss": loss, "gtloss": gtloss})
    dataset = pd.DataFrame(loss_list)
    std = dataset.loss.std()
    count = dataset.loss.count()
    unc = std / np.sqrt(count)
    print(f"Mean {dataset.loss.mean():.3e} Unc {unc:.3e}")
    return dataset
