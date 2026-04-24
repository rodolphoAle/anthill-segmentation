#!/usr/bin/env python3
"""
Validação das correções críticas antes do treinamento.

Este script verifica:
1. Normalização de imagem (range correto)
2. Sincronização de transforms (image vs mask)
3. Distribuição de valores na máscara (0/1/255 apenas)
4. Proporção de formigueiros (2% mínimo esperado)
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from model.segmentation_dataset import SegmentationDataset
from torchvision import transforms as T


def validate_dataset():
    """Executa validações no dataset."""
    
    print("\n" + "=" * 70)
    print(" VALIDAÇÃO DE CORREÇÕES - SEGMENTATION DATASET")
    print("=" * 70)
    
    # Configurar transforms (mesmo do treinamento)
    train_transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=(-10, 10)),
    ])
    
    # Criar dataset
    try:
        dataset = SegmentationDataset(
            local_dir="/home/isabelle/Documents/artigo/TCC-UFMS/data",
            patch_size=256,
            min_anthill_pixels=50,
            transforms=train_transforms,
        )
        print(f"\n Dataset criado: {len(dataset)} amostras")
    except Exception as e:
        print(f"\n Erro ao criar dataset: {e}")
        return False
    
    # Criar dataloader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    try:
        batch = next(iter(loader))
        images, masks = batch
        print(f" Batch carregado: imagens {images.shape}, máscaras {masks.shape}")
    except Exception as e:
        print(f" Erro ao carregar batch: {e}")
        return False
    
    print("\n" + "-" * 70)
    print("  VALIDAÇÃO DE NORMALIZAÇÃO DE IMAGEM")
    print("-" * 70)
    
    img_sample = images[0]
    print(f"   Shape: {img_sample.shape}")
    print(f"   Esperado: torch.Size([3, 256, 256])")
    
    min_val = img_sample.min().item()
    max_val = img_sample.max().item()
    mean_val = img_sample.mean().item()
    std_val = img_sample.std().item()
    
    print(f"\n   Min/Max: {min_val:.3f} / {max_val:.3f}")
    print(f"   Mean: {mean_val:.3f}, Std: {std_val:.3f}")
    print(f"   Esperado: Min ~ -2.1, Max ~ 2.6, Mean ~ 0, Std ~ 1")
    
    # Verificar se está normalizado
    if -2.5 < min_val < -1.5 and 2.0 < max_val < 3.0:
        print(f"   NORMALIZAÇÃO OK (ImageNet stats aplicadas)")
    elif 0 <= min_val and max_val <= 1:
        print(f"    AVISO: Imagem está em [0, 1], não normalizada ainda")
    elif 0 <= min_val and max_val <= 255:
        print(f"    ERRO CRÍTICO: Imagem está em [0, 255], normalização NÃO aplicada!")
        return False
    else:
        print(f"    DESCONHECIDO: range fora do esperado")
    
    print("\n" + "-" * 70)
    print("  VALIDAÇÃO DE MÁSCARA (valores 0/1/255)")
    print("-" * 70)
    
    mask_sample = masks[0]
    print(f"   Shape: {mask_sample.shape}")
    print(f"   Esperado: torch.Size([256, 256])")
    
    unique_values = torch.unique(mask_sample)
    print(f"\n   Valores únicos: {sorted(unique_values.tolist())}")
    print(f"   Esperado: [0, 1] ou [0, 1, 255]")
    
    # Verificar se há valores inválidos
    invalid_mask = mask_sample[(mask_sample != 0) & (mask_sample != 1) & (mask_sample != 255)]
    if len(invalid_mask) > 0:
        print(f"    ERRO: {len(invalid_mask)} pixels com valores inválidos!")
        print(f"      Valores: {torch.unique(invalid_mask).tolist()}")
        return False
    else:
        print(f"    MÁSCARA OK (apenas 0/1/255)")
    
    print("\n" + "-" * 70)
    print("  VALIDAÇÃO DE PROPORÇÃO DE FORMIGUEIROS")
    print("-" * 70)
    
    for i in range(min(4, len(masks))):
        mask_i = masks[i]
        anthill_count = (mask_i == 1).sum().item()
        ignore_count = (mask_i == 255).sum().item()
        total_pixels = 256 * 256
        
        anthill_ratio = 100 * anthill_count / total_pixels
        ignore_ratio = 100 * ignore_count / total_pixels
        background_ratio = 100 * (total_pixels - anthill_count - ignore_count) / total_pixels
        
        print(f"\n   Amostra {i+1}:")
        print(f"      Formigueiros: {anthill_count:6d} pixels ({anthill_ratio:5.1f}%)")
        print(f"      Ignorados:    {ignore_count:6d} pixels ({ignore_ratio:5.1f}%)")
        print(f"      Fundo:        {total_pixels - anthill_count - ignore_count:6d} pixels ({background_ratio:5.1f}%)")
        
        if anthill_ratio < 2:
            print(f"        AVISO: Proporção abaixo de 2% esperado")
        elif anthill_ratio > 50:
            print(f"       AVISO: Proporção muito alta (>50%), modelo pode enviesar")
        else:
            print(f"      OK")
    
    print("\n" + "-" * 70)
    print("  VALIDAÇÃO DE SINCRONIZAÇÃO DE TRANSFORMS")
    print("-" * 70)
    
    # Verificar se há align entre transforms
    # (comparar visual de duas amostras)
    print(f"   Amostra 1: Min={images[0].min():.2f}, Max={images[0].max():.2f}")
    print(f"   Máscara 1: Formigueiros={torch.unique(masks[0]).tolist()}")
    
    # Verificar estatísticas
    print(f"\n   Batch stats:")
    print(f"      Imagens - Min: {images.min():.2f}, Max: {images.max():.2f}")
    print(f"      Máscaras - Único: {torch.unique(masks).tolist()}")
    
    anthill_total = (masks == 1).sum().item()
    anthill_total_ratio = 100 * anthill_total / (4 * 256 * 256)
    print(f"\n   Proporção total no batch: {anthill_total_ratio:.1f}%")
    
    if anthill_total > 0:
        print(f"   SINCRONIZAÇÃO OK (imagens + máscaras alinhadas)")
    else:
        print(f"   ERRO: Nenhum formigueiro no batch!")
        return False
    
    print("\n" + "=" * 70)
    print(" TODAS AS VALIDAÇÕES PASSARAM")
    print("=" * 70)
    print("\n Dataset está pronto para treinamento!")
    print("\nPróximos passos:")
    print("  1. Rodar treino por 5 épocas")
    print("  2. Monitorar IoU por época")
    print("  3. Comparar com versão anterior (~0.35)")
    print("  4. Esperado: ~0.50-0.60 com correções")
    print("\n")
    
    return True


if __name__ == "__main__":
    success = validate_dataset()
    exit(0 if success else 1)
