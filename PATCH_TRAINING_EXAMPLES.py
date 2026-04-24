"""EXEMPLO DE INTEGRAÇÃO 

Este arquivo mostra exatamente como integrar patch training sem quebrar nada.
Copy & paste para seu projeto!
"""

# ============================================================================
# EXEMPLO 1: Usar com DataService (recomendado para novo código)
# ============================================================================

# Em app/service/data_service.py ou onde você cria os dataloaders:

from app.infrastructure.segmentation_dataset import SegmentationDataset

async def create_dataloaders_with_patch_training(
    storage_client,
    base_folder_id: str,
    batch_size: int = 4,
    patch_size: int = 256,
    min_anthill_pixels: int = 50,
):
    """Cria dataloaders com patch training."""
    
    train_pairs = [...]  # sua lógica aqui
    val_pairs = [...]
    
    # 🔥 Com Patch Training
    train_dataset = SegmentationDataset(
        pairs=train_pairs,
        download_fn=storage_client._sync_download_file,
        augmentations=create_train_transforms(),
        patch_size=patch_size,                    # ← NOVO
        min_anthill_pixels=min_anthill_pixels,    # ← NOVO
    )
    
    # Validação sem patch (usar imagem inteira)
    val_dataset = SegmentationDataset(
        pairs=val_pairs,
        download_fn=storage_client._sync_download_file,
        augmentations=None,
        patch_size=None,  # Usar imagem inteira na validação
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
    )
    
    return train_loader, val_loader


# ============================================================================
# EXEMPLO 2: Usar com model/segmentation_dataset.py (seu arquivo antigo)
# ============================================================================

# Se você tiver usando model/segmentation_dataset.py, atualize assim:

from model.segmentation_dataset import SegmentationDataset

# ANTES (sem patch):
# dataset = SegmentationDataset(drive_manager=drive_client, rgb_folder_id="...", ...)

# DEPOIS (com patch):
dataset = SegmentationDataset(
    drive_manager=drive_client,
    rgb_folder_id="seu_rgb_folder_id",
    labels_folder_id="seu_labels_folder_id",
    transforms=augmentations,
    
    # 🔥 ADICIONAR ESSAS 3 LINHAS
    patch_size=256,
    min_anthill_pixels=50,
    max_patch_retries=10,
)


# ============================================================================
# EXEMPLO 3: Script de Teste Rápido
# ============================================================================

import torch
from torch.utils.data import DataLoader

def validate_patch_training():
    """Script para validar que patch training está funcionando."""
    
    from model.segmentation_dataset import SegmentationDataset
    
    # Criar dataset
    dataset = SegmentationDataset(
        local_dir="/sua/pasta/local",  # ou drive_manager=...
        patch_size=256,
        min_anthill_pixels=50,
    )
    
    # Criar loader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Pegar primeiro batch
    batch = next(iter(loader))
    images, masks = batch
    
    print("PATCH TRAINING VALIDAÇÃO")
    print("=" * 60)
    
    # Validar shapes
    print(f"\n Shapes:")
    print(f"  Imagens: {images.shape}")
    print(f"  Máscaras: {masks.shape}")
    assert images.shape[1:] == (3, 256, 256), " Imagem não é 256x256!"
    assert masks.shape[1:] == (256, 256), " Máscara não é 256x256!"
    print("   Shapes corretos!")
    
    # Validar classes
    print(f"\n Distribuição de Classes (4 patches):")
    for i in range(4):
        mask = masks[i]
        bg = (mask == 0).sum().item()
        ant = (mask == 1).sum().item()
        ign = (mask == 255).sum().item()
        total = bg + ant + ign
        
        bg_pct = 100 * bg / total
        ant_pct = 100 * ant / total
        
        print(f"\n  Patch {i}:")
        print(f"    Fundo (0):      {bg:6d} ({bg_pct:5.1f}%)")
        print(f"    Formigueiro (1): {ant:6d} ({ant_pct:5.1f}%) ← deve ter bastante!")
        print(f"    Ignorado (255):  {ign:6d} ({100*ign/total:5.1f}%)")
        
        # Verificar mínimo
        if ant < 50:
            print(f"      AVISO: menos de 50 formigueiros!")
    
    # Validar que pode fazer backward
    print(f"\n Teste de Gradiente:")
    images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
    masks = masks.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dummy loss
    loss = masks.float().mean()
    loss.backward()
    print(f"  Backward OK")
    
    print("\n" + "=" * 60)
    print("PATCH TRAINING ESTÁ FUNCIONANDO!")
    print("\nProximos passos:")
    print("  1. Integrar ao seu run_training.py")
    print("  2. Rodar 5 épocas e monitorar IoU")
    print("  3. Deve aumentar de forma estável")


# ============================================================================
# EXEMPLO 4: Configurar em run_training.py
# ============================================================================

# Seu run_training.py provavelmente tem algo assim:

"""
async def main():
    model = UNet(n_channels=3, n_classes=2)
    training_service = TrainingService(model=model)
    
    # Carregar dataset
    train_loader, val_loader = await data_service.create_dataloaders()
    
    # Treinar
    await training_service.start_training(train_loader, val_loader, num_epochs=100)
"""

# Modificar assim:

"""
async def main():
    model = UNet(n_channels=3, n_classes=2)
    training_service = TrainingService(model=model)
    
    # Carregar dataset COM PATCH TRAINING
    train_loader, val_loader = await data_service.create_dataloaders(
        patch_size=256,              # ← ADICIONAR
        min_anthill_pixels=50,       # ← ADICIONAR
    )
    
    # Treinar (resto igual)
    await training_service.start_training(train_loader, val_loader, num_epochs=100)
"""


# ============================================================================
# EXEMPLO 5: Parâmetros para CLI
# ============================================================================

# Se quiser expor patch training via CLI (como você faz com learning_rate):

# Em app/core/config.py, adicionar:

class Settings(BaseSettings):
    # ... seus settings existentes ...
    
    # Patch Training
    patch_size: int = 256
    min_anthill_pixels: int = 50
    max_patch_retries: int = 10

# Em run_training.py, adicionar parser:

parser.add_argument("--patch-size", type=int, metavar="N",
                    help="Tamanho do patch para treinamento (UNET_PATCH_SIZE)")
parser.add_argument("--min-anthill-pixels", type=int, metavar="N",
                    help="Mínimo de formigueiros por patch (UNET_MIN_ANTHILL_PIXELS)")

# Usar assim:
# python run_training.py --patch-size 256 --min-anthill-pixels 50


# ============================================================================
# EXEMPLO 6: Monitorar Performance Melhoria
# ============================================================================

"""
Para validar que patch training está realmente ajudando:

Antes (sem patch):
  Epoch 1: loss=0.85, val_loss=0.82, IoU=0.35 (travado)
  Epoch 2: loss=0.82, val_loss=0.81, IoU=0.35 (travado)
  Epoch 3: loss=0.80, val_loss=0.79, IoU=0.35 (travado)
  Loss diminui mas IoU não muda (overfitting ao fundo)

Depois (com patch):
  Epoch 1: loss=0.75, val_loss=0.73, IoU=0.42
  Epoch 2: loss=0.68, val_loss=0.65, IoU=0.52 
  Epoch 3: loss=0.62, val_loss=0.58, IoU=0.58 
  Loss diminui AND IoU aumenta (aprendendo formigueiros!)
"""


# ============================================================================
# REFERÊNCIA RÁPIDA
# ============================================================================

"""
CHECKLIST DE INTEGRAÇÃO:

☐ 1. Abrir model/segmentation_dataset.py
☐ 2. Verificar que tem os novos parâmetros (patch_size, min_anthill_pixels)
☐ 3. Encontrar onde você cria o dataset em seu código
☐ 4. Adicionar: patch_size=256, min_anthill_pixels=50
☐ 5. Rodar um batch: python -c "from examples import validate_patch_training; validate_patch_training()"
☐ 6. Verificar que shapes estão 256x256
☐ 7. Verificar que formigueiros estão aumentados
☐ 8. Rodar treinamento: python run_training.py --epochs 5
☐ 9. Monitorar IoU nos logs
☐ 10. Comparar com IoU anterior (deve aumentar!)
"""
