import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, drive_manager=None, rgb_folder_id=None, labels_folder_id=None,
                 transforms=None, local_dir=None):

        self.transforms = transforms
        self.local_dir = local_dir
        self.drive_manager = drive_manager

        self.filenames = []
        self.labels_map = {}

        if local_dir:
            rgb_dir = os.path.join(local_dir, 'rgb')
            label_dir = os.path.join(local_dir, 'labels')

            rgb_files = sorted(os.listdir(rgb_dir))
            label_files = sorted(os.listdir(label_dir))

            label_dict = {l.split('.')[0]: l for l in label_files}

            for rgb in rgb_files:
                key = rgb.split('.')[0]
                if key in label_dict:
                    self.filenames.append(os.path.join(rgb_dir, rgb))
                    self.labels_map[rgb] = os.path.join(label_dir, label_dict[key])

        elif drive_manager:
            rgb_files = drive_manager.list_files_in_folder(rgb_folder_id)
            label_files = drive_manager.list_files_in_folder(labels_folder_id)

            label_dict = {l['name'].split('.')[0]: l['id'] for l in label_files}

            for rgb in rgb_files:
                key = rgb['name'].split('.')[0]
                if key in label_dict:
                    self.filenames.append(rgb)
                    self.labels_map[rgb['id']] = label_dict[key]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.local_dir:
            rgb_path = self.filenames[idx]
            label_path = self.labels_map[os.path.basename(rgb_path)]

            image = Image.open(rgb_path).convert("RGB")
            mask = Image.open(label_path)

        else:
            rgb_file = self.filenames[idx]
            label_id = self.labels_map[rgb_file['id']]

            image = Image.open(self.drive_manager.download_file(rgb_file['id'])).convert("RGB")
            mask = Image.open(self.drive_manager.download_file(label_id))

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        mask = torch.tensor(np.array(mask), dtype=torch.long)
        mask = torch.clamp(mask, 0, 1)

        return image, mask
