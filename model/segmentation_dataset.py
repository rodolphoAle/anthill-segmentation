import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, drive_manager=None, rgb_folder_id=None, labels_folder_id=None, transforms=None, local_dir=None):
        self.transforms = transforms
        self.local_dir = local_dir
        self.filenames = []
        self.labels_map = {}
        if local_dir:
            # Carregar arquivos locais
            import os
            rgb_dir = os.path.join(local_dir, 'rgb')
            label_dir = os.path.join(local_dir, 'labels')
            rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.tif'))])
            label_files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith('.png')])
            for rgb in rgb_files:
                prefix = "_".join(rgb.split("_")[:4])
                matched_label = [l for l in label_files if l.startswith(prefix)]
                if matched_label:
                    self.filenames.append(os.path.join(rgb_dir, rgb))
                    self.labels_map[os.path.join(rgb_dir, rgb)] = os.path.join(label_dir, matched_label[0])
        elif drive_manager:
            # Carregar arquivos do Google Drive
            rgb_files = drive_manager.list_files_in_folder(rgb_folder_id, extensions=['.png', '.jpg', '.jpeg', '.tif'])
            label_files = drive_manager.list_files_in_folder(labels_folder_id, extensions=['.png'])
            for rgb in rgb_files:
                prefix = "_".join(rgb['name'].split("_")[:4])
                matched_label = [l for l in label_files if l['name'].startswith(prefix)]
                if matched_label:
                    self.filenames.append(rgb)
                    self.labels_map[rgb['id']] = matched_label[0]['id']

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.local_dir:
            rgb_path = self.filenames[idx]
            label_path = self.labels_map[rgb_path]
            image = Image.open(rgb_path).convert("RGB")
            mask = Image.open(label_path)
        else:
            rgb_file = self.filenames[idx]
            label_file_id = self.labels_map[rgb_file['id']]
            image = Image.open(self.drive_manager.download_file(rgb_file['id'])).convert("RGB")
            mask = Image.open(self.drive_manager.download_file(label_file_id))
        if self.transforms:
            image, mask = self.transforms(image, mask)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        mask = torch.clamp(mask, 0, 1)
        return image, mask
