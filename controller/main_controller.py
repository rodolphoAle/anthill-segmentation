import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model.google_drive_manager import GoogleDriveManager
from model.model_manager import ModelManager
from model.segmentation_dataset import SegmentationDataset
from model.unet import UNet

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


class MainController:
    def __init__(self):
        self.drive = GoogleDriveManager()

    def baixar_dados(self, folder_id, destino):
        arquivos = self.drive.list_files_in_folder(folder_id)
        for arq in arquivos:
            self.drive.download_file(arq['id'], destination_path=f"{destino}/{arq['name']}")

    def run_training(self):
        dataset_train = SegmentationDataset(local_dir="data/train", transforms=transform)
        dataset_val = SegmentationDataset(local_dir="data/val", transforms=transform)

        train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=4)

        model = UNet()
        manager = ModelManager(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        manager.train(train_loader, val_loader, criterion, optimizer)


if __name__ == "__main__":
    controller = MainController()
    controller.run_training()
