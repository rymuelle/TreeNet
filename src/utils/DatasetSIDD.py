import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas as pd

# ------------------------------------------------------------------
# GLOBAL IMAGE CACHE
# maps: noisy_path → numpy array, gt_path → numpy array
# ------------------------------------------------------------------
_IMAGE_CACHE = {}
_CROP_SIZE_CACHE = {}   # store cropped versions keyed by (path, crop_size)


def load_and_crop_once(path, crop_size):
    """
    Loads an image from disk once, applies a center crop once,
    and returns a numpy array.
    """

    key = (path, crop_size)

    # Return cached version
    if key in _CROP_SIZE_CACHE:
        return _CROP_SIZE_CACHE[key]

    # Load original (uncropped) only once
    if path not in _IMAGE_CACHE:
        img = Image.open(path)
        img.load()                     
        img = img.convert("RGB")    
        _IMAGE_CACHE[path] = img
    else:
        img = _IMAGE_CACHE[path]

    # Compute crop
    w, h = img.size
    left   = (w - crop_size) // 2
    top    = (h - crop_size) // 2
    right  = left + crop_size
    bottom = top + crop_size

    cropped = np.asarray(img.crop((left, top, right, bottom))).copy()

    _CROP_SIZE_CACHE[key] = cropped
    return cropped


# ------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------
class DatasetSIDD(Dataset):
    def __init__(self, scene_folders, patch_size=256, crop_size=2560,
                 validation=False, transform=None, max_images=0,
                 supress_tqdm=True):

        self.patch_size = patch_size
        self.validation = validation
        self.transform = transform
        self.supress_tqdm = supress_tqdm

        # list of (noisy_path, gt_path)
        self.samples = []
        for scene in scene_folders:
            noisy_imgs = sorted(glob.glob(os.path.join(scene, "*NOISY_SRGB_*.PNG")))
            gt_imgs    = sorted(glob.glob(os.path.join(scene, "*GT_SRGB_*.PNG")))
            for n, g in zip(noisy_imgs, gt_imgs):
                self.samples.append((n, g))

        # Preload & crop all images once (cached globally!)
        print(f"Preparing {'validation' if validation else 'training'} data...")
        self.noisy_img = []
        self.gt_img = []
        self.image_name = []

        for noisy_path, gt_path in tqdm(self.samples, disable=self.supress_tqdm):
            noisy = load_and_crop_once(noisy_path, crop_size)
            clean = load_and_crop_once(gt_path, crop_size)

            self.noisy_img.append(noisy)
            self.gt_img.append(clean)
            self.image_name.append((noisy_path, gt_path))

            if max_images and len(self.noisy_img) >= max_images:
                break

    def __len__(self):
        return len(self.noisy_img)

    def __getitem__(self, idx):
        noisy = transforms.ToTensor()(self.noisy_img[idx])
        clean = transforms.ToTensor()(self.gt_img[idx])

        ps = self.patch_size
        _, H, W = noisy.shape

        # Choose crop location
        if self.validation:
            top = (H - ps) // 2
            left = (W - ps) // 2
        else:
            top = random.randint(0, H - ps)
            left = random.randint(0, W - ps)

        noisy = noisy[:, top:top+ps, left:left+ps]
        clean = clean[:, top:top+ps, left:left+ps]

        # Augmentations
        if not self.validation:
            if random.random() < 0.5:
                noisy = F.hflip(noisy); clean = F.hflip(clean)
            if random.random() < 0.5:
                noisy = F.vflip(noisy); clean = F.vflip(clean)
            k = random.randint(0, 3)
            if k:
                noisy = torch.rot90(noisy, k, (1,2))
                clean = torch.rot90(clean, k, (1,2))

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
