import os
import time
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from model.google_drive_manager import GoogleDriveManager
from model.segmentation_dataset import SegmentationDataset
from model.unet import UNet
from model.model_manager import ModelManager

# Transformações
data_transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.PILToTensor(),
])

data_transforms_val = transforms.Compose([
    transforms.PILToTensor(),
])

if __name__ == '__main__':
    # Escolha o modo: 'online' (Google Drive) ou 'local' (arquivos já baixados)
    modo = 'online'  # ou 'local'
    if modo == 'online':
        drive_manager = GoogleDriveManager()
        base_folder_id = '1slS6V7OWBaBny7v94K3Vx9eGHp7lph91'
        train_folder_id = drive_manager.get_folder_id('treino', base_folder_id)
        val_folder_id = drive_manager.get_folder_id('validacao', base_folder_id)
        train_rgb_folder_id = drive_manager.get_folder_id('rgb', train_folder_id)
        train_labels_folder_id = drive_manager.get_folder_id('labels', train_folder_id)
        val_rgb_folder_id = drive_manager.get_folder_id('rgb', val_folder_id)
        val_labels_folder_id = drive_manager.get_folder_id('labels', val_folder_id)
        image_dataset_train = SegmentationDataset(drive_manager=drive_manager, rgb_folder_id=train_rgb_folder_id, labels_folder_id=train_labels_folder_id, transforms=data_transforms_train)
        image_dataset_val = SegmentationDataset(drive_manager=drive_manager, rgb_folder_id=val_rgb_folder_id, labels_folder_id=val_labels_folder_id, transforms=data_transforms_val)
    else:
        image_dataset_train = SegmentationDataset(local_dir='data/train', transforms=data_transforms_train)
        image_dataset_val = SegmentationDataset(local_dir='data/val', transforms=data_transforms_val)

    # Dataloaders
    dataloader_train = DataLoader(image_dataset_train, batch_size=4, shuffle=True, num_workers=2)
    dataloader_val = DataLoader(image_dataset_val, batch_size=4, shuffle=False, num_workers=2)

    # Função para visualizar imagens
    def visualize_images(dataloader):
        print("Visualizando imagens do dataloader...")
        try:
            inputs, labels = next(iter(dataloader))
            print(f"Shape de inputs: {inputs.shape}")
            print(f"Shape de labels: {labels.shape}")
            
            fig, axs = plt.subplots(4, 2, figsize=(8, 16))
            for i in range(4):
                rgb = np.transpose(inputs[i].numpy(), (1, 2, 0)).astype(np.uint8)
                axs[i, 0].imshow(rgb)
                axs[i, 0].set_title('RGB')
                axs[i, 1].imshow(labels[i].squeeze(), cmap='gray')
                axs[i, 1].set_title('Mask')
            plt.show()
        except Exception as e:
            print(f"Erro ao visualizar imagens: {str(e)}")


    # Configuração do dispositivo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Modelo, critério e otimizador
    model = UNet()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Gerenciador de modelo
    model_manager = ModelManager(model, device=device)

    # Treinamento usando ModelManager
    print("Iniciando treinamento...")
    model_manager.train(dataloader_train, criterion, optimizer, num_epochs=20)

    # Avaliação opcional
    val_loss = model_manager.evaluate(dataloader_val, criterion)
    print(f"Val Loss final: {val_loss:.4f}")

    # Salvar o modelo
    model_save_path = 'u_net.pth'
    model_manager.save(model_save_path)
    print(f"Modelo salvo em: {model_save_path}")

    # Função para carregar e testar uma imagem
    def open_img(drive_manager, file_id, device):
        print(f"Carregando imagem com ID: {file_id}...")
        stream = drive_manager.download_file(file_id)
        img = Image.open(stream)
        img = data_transforms_val(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        print(f"Imagem carregada e preprocessada com shape: {img.shape}")
        return img

    # Função para salvar o modelo no Google Drive
    from googleapiclient.http import MediaFileUpload
    def save_model_to_drive(service, model_path, drive_folder_id):
        print(f"Salvando modelo em {model_path} no Google Drive na pasta ID {drive_folder_id}...")
        try:
            file_metadata = {'name': os.path.basename(model_path), 'parents': [drive_folder_id]}
            media = MediaFileUpload(model_path)
            file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print(f"Modelo salvo no Google Drive com ID: {file.get('id')}")
        except Exception as e:
            print(f"Erro ao salvar o modelo no Google Drive: {str(e)}")