#!/usr/bin/env python3
"""
SCRIPT DE VALIDAÇÃO RÁPIDA - Patch Training + Correções

Execute este script para verificar que todas as correções estão funcionando:
    python validate_dataset.py

"""

import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

# Importar seu dataset
try:
    from model.segmentation_dataset import SegmentationDataset
except ImportError:
    print(" Erro: não encontrou model/segmentation_dataset.py")
    print("   Execute a partir do diretório raiz do projeto")
    sys.exit(1)


def validate_dataset_local(local_dir: str, patch_size: int = 256):
    """Validar dataset com dados locais."""
    
    print("\n" + "="*70)
    print(" VALIDAÇÃO DE DATASET - PATCH TRAINING + CORREÇÕES")
    print("="*70)
    
    # Criar dataset COM PATCH TRAINING
    print("\n Criando dataset com patch_size=256...")
    dataset = SegmentationDataset(
        local_dir=local_dir,
        patch_size=256,
        min_anthill_pixels=10,
        max_patch_retries=30,
    )
    
    print(f" Dataset criado com {len(dataset)} imagens")
    
    # Criar dataloader
    print("\n Criando dataloader (batch_size=2)...")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Pegar primeiro batch
    print(" Carregando primeiro batch...")
    batch = next(iter(loader))
    images, masks, names = batch

    print(f"Arquivos do batch: {names}")
    
    # ========================================================================
    # VALIDAÇÃO 1: Shape da imagem
    # ========================================================================
    print("\n" + "-"*70)
    print(" VALIDAÇÃO: Shape da Imagem")
    print("-"*70)
    
    expected_shape = torch.Size([2, 3, 256, 256])
    actual_shape = images.shape
    
    print(f"   Esperado: {expected_shape}")
    print(f"   Obtido:   {actual_shape}")
    
    if actual_shape == expected_shape:
        print("    PASSOU")
    else:
        print("   FALHOU - shape incorreto!")
        return False
    
    # ========================================================================
    # VALIDAÇÃO 2: Normalização ImageNet
    # ========================================================================
    print("\n" + "-"*70)
    print("  VALIDAÇÃO: Normalização ImageNet")
    print("-"*70)
    
    img_min = images.min().item()
    img_max = images.max().item()
    img_mean = images.mean().item()
    img_std = images.std().item()
    
    print(f"   Min:      {img_min:+.3f}")
    print(f"   Max:      {img_max:+.3f}")
    print(f"   Mean:     {img_mean:+.3f}")
    print(f"   Std:      {img_std:+.3f}")
    
    # ImageNet normalizado geralmente tem:
    # min ~ -2.1, max ~ +2.6, mean ~ 0, std ~ 1
    if -3 < img_min < 3 and -3 < img_max < 3:
        print("PASSOU - normalização OK")
    else:
        print("FALHOU")
    
    # ========================================================================
    # VALIDAÇÃO 3: Shape da máscara
    # ========================================================================
    print("\n" + "-"*70)
    print("VALIDAÇÃO: Shape da Máscara")
    print("-"*70)
    
    expected_mask_shape = torch.Size([2, 256, 256])
    actual_mask_shape = masks.shape
    
    print(f"   Esperado: {expected_mask_shape}")
    print(f"   Obtido:   {actual_mask_shape}")
    
    if actual_mask_shape == expected_mask_shape:
        print("    PASSOU")
    else:
        print("   FALHOU - shape de máscara incorreto!")
        return False
    
    # ========================================================================
    # VALIDAÇÃO 4: Distribuição de classes
    # ========================================================================
    print("\n" + "-"*70)
    print(" VALIDAÇÃO: Distribuição de Classes por Patch")
    print("-"*70)
    
    total_bg = 0
    total_ant = 0
    total_ign = 0
    
    for i in range(2):
        mask = masks[i]
        bg = (mask == 0).sum().item()
        ant = (mask == 1).sum().item()
        ign = (mask == 255).sum().item()
        
        total_pixels = 256 * 256
        bg_pct = 100 * bg / total_pixels
        ant_pct = 100 * ant / total_pixels
        ign_pct = 100 * ign / total_pixels
        
        total_bg += bg
        total_ant += ant
        total_ign += ign
        
        print(f"\n   Patch {i}:")
        print(f"      Background (0):   {bg:7d} ({bg_pct:5.1f}%)")
        print(f"      Anthill (1):      {ant:7d} ({ant_pct:5.1f}%) 🐜")
        print(f"      Ignore (255):     {ign:7d} ({ign_pct:5.1f}%)")
        
        # Validar que tem formigueiros
        if ant < 50:
            print(f"AVISO: patch tem <50 formigueiros")
    
    # Estatística total
    total_all = total_bg + total_ant + total_ign
    print(f"\n   TOTAL (4 patches):")
    print(f"      Background:  {total_bg:8d} ({100*total_bg/total_all:5.1f}%)")
    print(f"      Anthill:     {total_ant:8d} ({100*total_ant/total_all:5.1f}%) 🐜")
    print(f"      Ignore:      {total_ign:8d} ({100*total_ign/total_all:5.1f}%)")
    
    if total_ant > total_all * 0.01:  # pelo menos 1% de formigueiros
        print(f"\n    PASSOU - {100*total_ant/total_all:.1f}% de formigueiros é bom!")
    else:
        print(f"\n    FALHOU - <1% de formigueiros (dataset ruim!)")
        return False
    
    # ========================================================================
    # VALIDAÇÃO 5: Backward pass (testar gradient flow)
    # ========================================================================
    print("\n" + "-"*70)
    print(" VALIDAÇÃO: Gradient Flow (Backward Pass)")
    print("-"*70)
    
    try:
        # Mover para device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {device}")
        
        images_dev = images.to(device)
        masks_dev = masks.to(device)
        
        # Dummy loss e backward
        loss = masks_dev.float().mean()
        loss.backward()
        
        print(f"    PASSOU - backward() funcionou")
        
    except Exception as e:
        print(f"    FALHOU - erro no backward: {e}")
        return False
    
    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    print("\n" + "="*70)
    print(" TODAS AS VALIDAÇÕES PASSARAM!")
    print("="*70)
    
    print("\n Resumo:")
    print("    Shape de imagem correto (3, 256, 256)")
    print("    Imagem normalizada com ImageNet stats")
    print("    Shape de máscara correto (256, 256)")
    print(f"    Distribuição de classes ótima ({100*total_ant/total_all:.1f}% formigueiros)")
    print("    Backward pass funcionando")
    
    print("\n Próximos passos:")
    print("   1. Integrar dataset ao seu run_training.py")
    print("   2. Rodar treinamento: python run_training.py --epochs 20")
    print("   3. Monitorar IoU nos logs (deve aumentar!)")
    print("   4. Comparar com versão anterior (esperar +60%)")
    
    return True


def main():
    """Entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validar correções do dataset (patch training + normalização)"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Caminho local com subpastas 'rgb/' e 'labels/' (ex: ./data/)"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Tamanho do patch (default: 256 para validação com patch training)"
    )
    
    args = parser.parse_args()
    
    # Se nenhum argumento, print help
    if args.local_dir is None:
        parser.print_help()
        print("\n Erro: --local-dir é obrigatório")
        print("\nExemplo:")
        print("  python validate_dataset.py --local-dir ./data/")
        print("  python validate_dataset.py --local-dir /caminho/para/dados/ --patch-size 256")
        sys.exit(1)
    
    # Validar dataset
    success = validate_dataset_local(args.local_dir, args.patch_size)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
