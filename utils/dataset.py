



import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import scipy.io
import shutil
import keras

class CaltechDetection(Dataset):
    def __init__(self, root, category, image_set='train', transform=None, target_transform=None, transforms=None, download=False):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.category = category
        
        self.name_map = {
            "Faces": "Faces_2",
            "Faces_easy": "Faces_3",
            "Motorbikes": "Motorbikes_16",
            "airplanes": "Airplanes_Side_2",
            "cup": "cup",  
        }
        
        if download:
            self.download()
        
        self.image_dir = os.path.join(root, "101_ObjectCategories", self.category)
        self.annot_dir = os.path.join(root, "Annotations", self.name_map.get(self.category, self.category))
        
        self.image_paths = sorted([f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))])
        self.annot_paths = sorted([f for f in os.listdir(self.annot_dir) if os.path.isfile(os.path.join(self.annot_dir, f))])
        
        # Split dataset
        split_idx = int(len(self.image_paths) * 0.8)
        if image_set == 'train':
            self.image_paths = self.image_paths[:split_idx]
            self.annot_paths = self.annot_paths[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]
            self.annot_paths = self.annot_paths[split_idx:]
        
        self.data = []
        for img_path, annot_path in zip(self.image_paths, self.annot_paths):
            self.data.append((os.path.join(self.image_dir, img_path), os.path.join(self.annot_dir, annot_path)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, annot_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        annot = scipy.io.loadmat(annot_path)["box_coord"][0]
        
        # Original image size
        orig_w, orig_h = image.size
        
        # Convert to [xmin, ymin, xmax, ymax] format and scale to 224x224
        xmin = float((annot[2] / orig_w) * 224)
        ymin = float((annot[0] / orig_h) * 224)
        xmax = float((annot[3] / orig_w) * 224)
        ymax = float((annot[1] / orig_h) * 224)
        
        # Ensure coordinates are within [0, 224] range
        xmin = max(0, min(xmin, 224))
        ymin = max(0, min(ymin, 224))
        xmax = max(0, min(xmax, 224))
        ymax = max(0, min(ymax, 224))
        
        target = [xmin, ymin, xmax, ymax]
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        # Remove the extra dimension from the image tensor
        image = image.squeeze(0)
        
        return image, target

    def download(self):
        """Download and extract the Caltech101 dataset if it's not already present."""
        
        # Download the dataset
        path_to_downloaded_file = keras.utils.get_file(
            fname="caltech_101_zipped",
            origin="https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip",
            extract=True,
            archive_format="zip",
            cache_dir=self.root
        )
        download_base_dir = os.path.dirname(path_to_downloaded_file)

        # Extracting tar files found inside main zip file
        shutil.unpack_archive(
            os.path.join(download_base_dir, "caltech-101", "101_ObjectCategories.tar.gz"), 
            self.root
        )
        shutil.unpack_archive(
            os.path.join(download_base_dir, "caltech-101", "Annotations.tar"), 
            self.root
        )

        print(f"Dataset for {self.category} is ready!")

def read_caltech_dataset(path, category, download=False):
    T = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    caltech_train = CaltechDetection(path, category, image_set='train', transform=T, download=download)
    caltech_val = CaltechDetection(path, category, image_set='val', transform=T, download=False)
    
    return caltech_train, caltech_val