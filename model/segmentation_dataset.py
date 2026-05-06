import os
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image

class SegmentationDataset(Dataset):
    """
    Segmentation dataset with optional patch-based training.
    
     PATCH TRAINING:
    - Extrai patches (256x256) das imagens ao invés de usar imagem inteira
    - Filtra patches com quantidade mínima de formigueiros
    -  Resultado: muito mais formigueiro por pixel, menos fundo dominante
    
    Args:
        patch_size (int): Tamanho do patch em pixels. 
                         Se None, usa imagem inteira (compatível com código antigo).
                         Default: 256 para patch training.
        min_anthill_pixels (int): Mínimo de pixels de formigueiro no patch.
                                 Se patch tiver menos, tenta outro crop.
                                 Default: 50
        max_patch_retries (int): Máximo de tentativas para achar patch válido.
                                Default: 10
    """
    def __init__(self, drive_manager=None, rgb_folder_id=None, labels_folder_id=None,
                 transforms=None, local_dir=None, patch_size=256, min_anthill_pixels=10,
                 max_patch_retries=30):

        self.transforms = transforms
        self.local_dir = local_dir
        self.drive_manager = drive_manager
        self.patch_size = patch_size
        self.min_anthill_pixels = min_anthill_pixels
        self.max_patch_retries = max_patch_retries

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
    
    def _decode_mask_to_labels(self, mask_arr):
        """
        Decodifica máscara RGB para classes (0=background, 1=anthill, 255=ignore).
        
        Convenção RGB:
        - Red (R>150, G<100, B<100)   → class 1 (formigueiro)
        - Black (all < 50)             → class 0 (fundo)
        - White (all > 200)            → class 255 (não-rotulado, ignorado)
        - Else                         → class 255 (ignorado)
        """
        r, g, b = mask_arr[:, :, 0], mask_arr[:, :, 1], mask_arr[:, :, 2]
        
        is_anthill = (r > 150) & (g < 100) & (b < 100)
        is_background = (r < 50) & (g < 50) & (b < 50)
        
        label = np.full(mask_arr.shape[:2], 255, dtype=np.int64)
        label[is_background] = 0
        label[is_anthill] = 1
        
        return label

    def _extract_patch(self, image, mask, patch_size):
        """
        Extrai um patch (crop) de tamanho patch_size x patch_size.
        Tenta garantir que o patch tenha proporção mínima de formigueiros.
        
         MELHORIA: usa proporção (%) ao invés de pixel count fixo
           - mais robusto para diferentes resoluções
        
        Returns:
            (image_patch, mask_patch_decoded) ou (image_inteira, label_inteira) se <patch_size
        """
        w, h = image.size
        min_anthill_count = self.min_anthill_pixels
        
        # Se imagem é menor que patch_size, retornar imagem inteira
        if w < patch_size or h < patch_size:
            mask_arr = np.array(mask)
            if mask_arr.ndim == 3 and mask_arr.shape[2] == 3:
                label = self._decode_mask_to_labels(mask_arr)
            else:
                label = np.array(mask, dtype=np.int64)
            
            return image, label, None  # retorna None para indicar imagem inteira
        
        # Decodificar máscara UMA VEZ (otimização)
        mask_arr = np.array(mask)
        if mask_arr.ndim == 3 and mask_arr.shape[2] == 3:
            mask_decoded = self._decode_mask_to_labels(mask_arr)
        else:
            mask_decoded = np.array(mask, dtype=np.int64)
        
        best_patch = None
        best_anthill_count = 0
        
        # Tentar encontrar patch com formigueiros suficientes
        for attempt in range(self.max_patch_retries):
            x = random.randint(0, w - patch_size)
            y = random.randint(0, h - patch_size)
            
            # Crop das imagens
            image_patch = TF.crop(image, y, x, patch_size, patch_size)
            label_patch = mask_decoded[y:y+patch_size, x:x+patch_size]
            
            # Contar formigueiros
            anthill_count = (label_patch == 1).sum()
            
            # Se encontrou patch bom, retornar imediatamente
            if anthill_count >= min_anthill_count:
                return image_patch, label_patch, True
            
            # Guardar melhor patch encontrado (fallback)
            if anthill_count > best_anthill_count:
                best_anthill_count = anthill_count
                best_patch = (image_patch, label_patch)
        
        # Se não conseguiu patch ideal, retornar melhor encontrado
        if best_patch is not None:
            return best_patch[0], best_patch[1], False
        
        # Fallback final: retornar primeiro crop mesmo que ruim
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        image_patch = TF.crop(image, y, x, patch_size, patch_size)
        label_patch = mask_decoded[y:y+patch_size, x:x+patch_size]
        return image_patch, label_patch, False

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

        #  PATCH TRAINING: Extrair crop ao invés de usar imagem inteira
        if self.patch_size is not None:
            image, label, _ = self._extract_patch(image, mask, self.patch_size)
        else:
            mask_arr = np.array(mask)
            if mask_arr.ndim == 3 and mask_arr.shape[2] == 3:
                label = self._decode_mask_to_labels(mask_arr)
            else:
                label = np.array(mask, dtype=np.int64)

        #  Aplicar transforms CONJUNTOS em image + mask
        if self.transforms:
            image, mask_pil = self.transforms(image, Image.fromarray(label.astype(np.uint8)))
            label = np.array(mask_pil, dtype=np.int64)
         
            label = np.where(label > 1, 255, label).astype(np.int64)

        #normalização ImageNet 
        image_tensor = TF.to_tensor(image)  # (3, H, W) float in [0, 1]
        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        mask_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, mask_tensor, os.path.basename(rgb_path)