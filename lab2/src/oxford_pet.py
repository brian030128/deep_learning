import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None, num_augmentations=3, seed=1000):

        assert mode in {"train", "valid", "test"}

        self.seed = seed
        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits
        self.num_augmentations = num_augmentations if mode == "train" else 1
        
    def __len__(self):
        return len(self.filenames) * self.num_augmentations
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self.__getitem__(i) for i in indices]
        
        # Calculate which original image and which augmentation
        original_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations
        
        filename = self.filenames[original_idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")
        
        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)
        
        sample = dict(image=image, mask=mask, trimap=trimap)
        
        # Apply transforms based on augmentation index
        if self.transform is not None and (aug_idx > 0 or self.mode != "train"):
            # Use different random seeds for each augmentation
            random_seed = self.seed + idx
            sample = self.transform(**sample, seed=random_seed)
        
        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask
    


    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

class SimpleOxfordPetDataset(OxfordPetDataset):

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slicing and concatenate into a batch
            indices = range(*idx.indices(len(self)))
            samples = [self.__getitem__(i) for i in indices]
            return samples

        sample = super().__getitem__(idx)

        # Resize images and masks to a fixed size
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # Ensure masks are integers for class labels
        mask = mask.astype(np.int64)

        # Convert image format from HWC to CHW
        sample["image"] = np.moveaxis(image, -1, 0)  # Channels first
        sample["mask"] = mask  # Ensure mask is consistent
        sample["trimap"] = trimap

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)





def load_dataset(data_path, mode, transform=None):
    OxfordPetDataset.download(data_path)
    return SimpleOxfordPetDataset(data_path, mode,transform=transform if mode == "train" else None)



