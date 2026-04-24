#!/usr/bin/env python3
"""
TRAINING MONITOR - Captura métricas e predictions

Execute:
    python train_with_monitoring.py --epochs 5 --local-data

Saída:
    - training_metrics.csv (IoU, Loss por época)
    - sample_predictions/ (imagens + masks + predições)
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from PIL import Image
import torchvision.transforms.functional as TF

# Importar componentes
try:
    from app.domain.unet import UNet
    from model.segmentation_dataset import SegmentationDataset
    from app.core.config import settings
except ImportError as e:
    print(f" Erro ao importar: {e}")
    print("Execute a partir do diretório raiz do projeto")
    sys.exit(1)


class TrainingMonitor:
    """Monitor de treinamento com logging de métricas."""
    
    def __init__(self, output_dir="training_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.metrics = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n🖥️  Device: {self.device}")
        print(f"📁 Output dir: {self.output_dir}")
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Treina uma época e retorna loss médio."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def compute_iou(self, model, val_loader):
        """Computa IoU médio no validation set."""
        model.eval()
        
        intersection = np.zeros(2)  # [bg, anthill]
        union = np.zeros(2)
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.cpu().numpy()
                
                # Predição
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                # Computar IoU por classe
                for c in [0, 1]:
                    intersection[c] += ((preds == c) & (masks == c)).sum()
                    union[c] += ((preds == c) | (masks == c)).sum()
        
        # Computar IoU (Mean IoU)
        iou_per_class = intersection / (union + 1e-6)
        mean_iou = iou_per_class.mean()
        
        return mean_iou, iou_per_class
    
    def save_predictions(self, model, val_loader, epoch, num_samples=4):
        """Salva sample de predictions."""
        model.eval()
        
        pred_dir = self.output_dir / f"predictions_epoch{epoch+1}"
        pred_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                if batch_idx >= 1:  # Só 1 batch
                    break
                
                images_dev = images.to(self.device)
                outputs = model(images_dev)
                preds = outputs.argmax(dim=1)
                
                for i in range(min(num_samples, images.shape[0])):
                    # Desnormalizar imagem
                    img = images[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
                    img = (img * np.array([0.229, 0.224, 0.225]) + 
                           np.array([0.485, 0.456, 0.406]))
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    
                    # Masks
                    mask_gt = masks[i].cpu().numpy()
                    mask_pred = preds[i].cpu().numpy()
                    
                    # Salvar
                    Image.fromarray(img).save(pred_dir / f"sample_{i:02d}_image.png")
                    Image.fromarray((mask_gt * 127).astype(np.uint8)).save(
                        pred_dir / f"sample_{i:02d}_mask_gt.png"
                    )
                    Image.fromarray((mask_pred * 127).astype(np.uint8)).save(
                        pred_dir / f"sample_{i:02d}_mask_pred.png"
                    )
                    
                    print(f"    Saved sample {i+1} to {pred_dir}")
    
    def run(self, train_loader, val_loader, num_epochs=5, learning_rate=1e-4):
        """Roda treinamento completo."""
        print("\n" + "="*70)
        print("INICIANDO TREINAMENTO COM MONITORAMENTO")
        print("="*70)
        
        # Modelo e otimizador
        model = UNet(n_channels=3, n_classes=2)
        model = model.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        print(f"\n📊 Configuração:")
        print(f"   Épocas: {num_epochs}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Batch size train: {train_loader.batch_size}")
        print(f"   Batch size val: {val_loader.batch_size}")
        
        # Treinar
        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"📈 Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )
            
            # Validate
            print(f"\n  Validando...")
            mean_iou, iou_per_class = self.compute_iou(model, val_loader)
            
            # Salvar predictions
            print(f"  Salvando predictions...")
            self.save_predictions(model, val_loader, epoch)
            
            # Log
            self.metrics.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_iou': mean_iou,
                'iou_bg': iou_per_class[0],
                'iou_anthill': iou_per_class[1],
            })
            
            print(f"\n Época {epoch+1} Completa:")
            print(f"   Train Loss:     {train_loss:.4f}")
            print(f"   Val mIoU:       {mean_iou:.4f}")
            print(f"   IoU Background: {iou_per_class[0]:.4f}")
            print(f"   IoU Anthill:    {iou_per_class[1]:.4f}")
            
            # Scheduler
            scheduler.step(train_loss)
        
        # Salvar métricas em CSV
        df = pd.DataFrame(self.metrics)
        csv_path = self.output_dir / f"training_metrics_{self.timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        print("\n" + "="*70)
        print(" TREINAMENTO COMPLETO")
        print("="*70)
        print(f"\n Métricas salvas em: {csv_path}")
        print(f" Predictions em: {self.output_dir}/predictions_*")
        print(f"\nResumo Final:")
        print(df.to_string(index=False))
        
        return model, df


def main():
    parser = argparse.ArgumentParser(description="Train UNet with monitoring")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--local-data", action="store_true", help="Use local data")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Path to local data (requires 'rgb/' and 'labels/')")
    
    args = parser.parse_args()
    
    print(f"\n🔧 Configuração:")
    print(f"   Epochs: {args.epochs}")
    print(f"   LR: {args.lr}")
    print(f"   Batch size: {args.batch_size}")
    
    # Criar dataset
    if args.local_data:
        print(f"\n Carregando dados locais de: {args.data_dir}")
        
        if not os.path.exists(args.data_dir):
            print(f" Erro: {args.data_dir} não existe")
            sys.exit(1)
        
        # Transforms de treinamento
        train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=(-10, 10)),
        ])
        
        # Criar dataset
        dataset = SegmentationDataset(
            local_dir=args.data_dir,
            patch_size=256,
            min_anthill_pixels=50,
            max_patch_retries=10,
            transforms=train_transforms,
        )
        
        # Split: 80% train, 20% val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Validação sem patches (imagem inteira)
        val_dataset_no_patch = SegmentationDataset(
            local_dir=args.data_dir,
            patch_size=None,  # Sem patch training
            transforms=None,   # Sem augmentation
        )
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset_no_patch, batch_size=args.batch_size, shuffle=False, num_workers=2
        )
        
        print(f" Dataset carregado:")
        print(f"   Total: {len(dataset)} imagens")
        print(f"   Train: {len(train_dataset)} amostras")
        print(f"   Val: {len(val_dataset)} amostras")
        
    else:
        print(f" Erro: --local-data é obrigatório por enquanto")
        sys.exit(1)
    
    # Rodar treinamento
    monitor = TrainingMonitor()
    model, metrics = monitor.run(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
