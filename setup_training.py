#!/usr/bin/env python3
"""
SETUP PARA TREINAMENTO - Preparar ambiente e dados

Execute:
    python setup_training.py --check          # Apenas verificar
    python setup_training.py --download-data  # Baixar dados do Drive
"""

import os
import sys
import argparse
import torch
from pathlib import Path

def check_environment():
    """Verifica se o ambiente está pronto."""
    print("\n" + "="*70)
    print("🔍 VERIFICANDO AMBIENTE")
    print("="*70)
    
    checks = {}
    
    # 1. Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"\n✓ Python: {py_version}")
    checks['python'] = True
    
    # 2. PyTorch
    try:
        import torch
        print(f" PyTorch: {torch.__version__}")
        checks['torch'] = True
    except ImportError:
        print(f"✗ PyTorch: NÃO INSTALADO")
        checks['torch'] = False
    
    # 3. CUDA
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    if cuda_available:
        print(f" CUDA: Disponível (device: {device})")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"⚠ CUDA: NÃO DISPONÍVEL (usando CPU - será LENTO)")
    checks['cuda'] = cuda_available
    
    # 4. Dependências
    deps = ['numpy', 'pandas', 'pillow', 'loguru', 'pydantic', 'torchvision']
    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep}")
            checks[dep] = True
        except ImportError:
            print(f"✗ {dep}: NÃO INSTALADO")
            checks[dep] = False
    
    # 5. Arquivos críticos
    print(f"\n Verificando arquivos:")
    
    files_required = {
        'credentials.json': 'Google Drive credentials',
        'app/domain/unet.py': 'UNet model',
        'model/segmentation_dataset.py': 'Dataset class',
        'app/core/config.py': 'Configuration',
    }
    
    for file, desc in files_required.items():
        if os.path.exists(file):
            print(f"✓ {file} ({desc})")
            checks[file] = True
        else:
            print(f"✗ {file} ({desc}) - FALTANDO")
            checks[file] = False
    
    # 6. Dados
    print(f"\n📊 Verificando dados:")
    
    data_dir = Path("data")
    if data_dir.exists():
        rgb_dir = data_dir / "rgb"
        labels_dir = data_dir / "labels"
        
        if rgb_dir.exists() and labels_dir.exists():
            num_rgb = len(list(rgb_dir.glob("*")))
            num_labels = len(list(labels_dir.glob("*")))
            print(f"✓ data/rgb: {num_rgb} imagens")
            print(f"✓ data/labels: {num_labels} máscaras")
            checks['data'] = num_rgb > 0 and num_labels > 0
        else:
            print(f"⚠ data/ existe mas está vazio (rgb ou labels faltam)")
            checks['data'] = False
    else:
        print(f"⚠ data/ NÃO EXISTE (dados do Google Drive)")
        checks['data'] = False
    
    # 7. .env
    if os.path.exists('.env'):
        print(f"✓ .env configurado")
        checks['.env'] = True
    else:
        print(f"✗ .env FALTANDO")
        checks['.env'] = False
    
    # Resumo
    print("\n" + "-"*70)
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    print(f"\n Resultado: {passed}/{total} verificações passaram")
    
    if passed == total:
        print("\nAmbiente pronto para treinamento!")
        return True
    else:
        print("\n Algumas verificações falharam:")
        for check, result in checks.items():
            if not result:
                print(f"  - {check}")
        return False


def check_dataset():
    """Verifica o dataset."""
    print("\n" + "="*70)
    print("🔍 VERIFICANDO DATASET")
    print("="*70)
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print(f"\n data/ não existe")
        print(f"\nOpções:")
        print(f"1. Usar Google Drive (online)")
        print(f"   - Configure UNET_DATA_MODE=online no .env")
        print(f"   - Coloque credentials.json no diretório raiz")
        print(f"\n2. Usar dados locais")
        print(f"   - Crie data/rgb/ com imagens")
        print(f"   - Crie data/labels/ com máscaras")
        return False
    
    rgb_dir = data_dir / "rgb"
    labels_dir = data_dir / "labels"
    
    if not (rgb_dir.exists() and labels_dir.exists()):
        print(f" Faltam diretórios:")
        print(f"  rgb_dir: {rgb_dir.exists()}")
        print(f"  labels_dir: {labels_dir.exists()}")
        return False
    
    # Contar arquivos
    rgb_files = sorted(list(rgb_dir.glob("*")))
    label_files = sorted(list(labels_dir.glob("*")))
    
    print(f"\n✓ {len(rgb_files)} imagens em data/rgb/")
    print(f"✓ {len(label_files)} máscaras em data/labels/")
    
    if len(rgb_files) == 0 or len(label_files) == 0:
        print(f"\n Dataset vazio!")
        return False
    
    if len(rgb_files) != len(label_files):
        print(f"\n Número de imagens e máscaras diferente!")
        print(f"  Imagens: {len(rgb_files)}")
        print(f"  Máscaras: {len(label_files)}")
        return False
    
    print(f"\n Dataset verificado com sucesso!")
    print(f"   {len(rgb_files)} pares de imagem-máscara")
    
    return True


def show_next_steps():
    """Mostra próximos passos."""
    print("\n" + "="*70)
    print(" PRÓXIMOS PASSOS")
    print("="*70)
    
    print(f"\n1. Validar correções do dataset:")
    print(f"   python validate_fixes.py")
    
    print(f"\n2. Verificar o dataset:")
    print(f"   python setup_training.py --check-dataset")
    
    print(f"\n3. Rodar treinamento (5 épocas, monitorado):")
    print(f"   python train_with_monitoring.py --epochs 5 --local-data --data-dir data/")
    
    print(f"\n4. Depois do treino:")
    print(f"   - Verificar training_metrics_*.csv")
    print(f"   - Ver predictions_epoch* para samples")
    print(f"   - Comparar IoU com versão anterior (esperado +60%)")


def main():
    parser = argparse.ArgumentParser(description="Setup para treinamento")
    parser.add_argument("--check", action="store_true", help="Verificar ambiente")
    parser.add_argument("--check-dataset", action="store_true", help="Verificar dataset")
    parser.add_argument("--next-steps", action="store_true", help="Mostrar próximos passos")
    
    args = parser.parse_args()
    
    if not any([args.check, args.check_dataset, args.next_steps]):
        args.check = True  # Default
    
    if args.check:
        ok = check_environment()
        if not ok:
            print("\n Por favor, instale as dependências faltantes")
            sys.exit(1)
    
    if args.check_dataset:
        ok = check_dataset()
        if not ok:
            print("\n Dataset não está pronto")
            sys.exit(1)
    
    if args.next_steps:
        show_next_steps()
    
    print("\n")


if __name__ == "__main__":
    main()
