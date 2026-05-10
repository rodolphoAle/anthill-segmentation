"""Serviço responsável pela preparação de datasets e criação de DataLoaders.

Responsabilidades:
- baixar datasets remotos;
- preparar diretórios locais;
- criar datasets de treino e validação;
- criar DataLoaders PyTorch;
- suportar datasets locais e streaming.

O serviço depende de StorageClientProtocol,
mantendo desacoplamento entre regra de negócio
e implementação de armazenamento.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from app.core.config import settings
from app.core.exceptions import DatasetNotFoundError, FolderNotFoundError
from app.domain.protocols import StorageClientProtocol
from app.infrastructure.augmentations import (
    create_image_only_transforms,
    create_train_transforms,
)
from app.infrastructure.segmentation_dataset import SegmentationDataset
from app.infrastructure.streaming_dataset import StreamingSegmentationDataset


def _collate_with_names(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Agrupa batches mantendo nomes dos arquivos."""

    # Empilha imagens.
    images = torch.stack([item[0] for item in batch])

    # Empilha máscaras.
    masks = torch.stack([item[1] for item in batch])

    # Mantém nomes em lista separada.
    names = [item[2] for item in batch]

    return images, masks, names


# Serviço principal

class DataService:
    """Serviço responsável por datasets e DataLoaders."""

    def __init__(
        self,
        storage_client: StorageClientProtocol | None = None,
    ) -> None:
        # Cliente de armazenamento remoto.
        self._storage_client = storage_client

    # Download remoto

    async def _resolve_subfolder_id(
        self,
        parent_id: str,
        *names: str,
    ) -> str:
        """Resolve IDs de subpastas remotas."""

        if self._storage_client is None:
            raise DatasetNotFoundError(
                "No storage client configured for online mode"
            )

        current_id = parent_id

        # Percorre estrutura de pastas.
        for name in names:

            folder_id = await self._storage_client.get_folder_id(
                name,
                current_id,
            )

            # Pasta não encontrada.
            if folder_id is None:
                raise FolderNotFoundError(
                    f"Folder '{name}' not found under parent '{current_id}'"
                )

            current_id = folder_id

        return current_id

    async def _download_folder_contents(
        self,
        folder_id: str,
        local_dir: Path,
        extensions: list[str],
    ) -> int:
        """Baixa arquivos remotos para diretório local."""

        if self._storage_client is None:
            raise DatasetNotFoundError(
                "No storage client configured for online mode"
            )

        # Cria diretório local.
        local_dir.mkdir(parents=True, exist_ok=True)

        # Lista arquivos remotos.
        files = await self._storage_client.list_files(
            folder_id,
            extensions=extensions,
        )

        downloaded = 0

        for file_info in files:

            dest = local_dir / file_info["name"]

            # Ignora arquivos já existentes.
            if dest.exists():
                continue

            # Realiza download.
            await self._storage_client.download_file(
                file_info["id"],
                destination_path=str(dest),
            )

            downloaded += 1

        logger.info(
            "Downloaded {} new files to {} ({} total on disk)",
            downloaded,
            local_dir,
            len(files),
        )

        return downloaded

    async def download_dataset_from_drive(
        self,
        base_folder_id: str,
        destination: str | Path,
    ) -> dict[str, Path]:
        """Baixa dataset completo do Google Drive."""

        destination = Path(destination)

        image_extensions = [
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
        ]

        paths: dict[str, Path] = {}

        # Processa treino e validação.
        for split in ("treino", "validacao"):

            for subfolder in ("rgb", "labels"):

                # Resolve pasta remota.
                remote_id = await self._resolve_subfolder_id(
                    base_folder_id,
                    split,
                    subfolder,
                )

                # Diretório local.
                local_dir = destination / split / subfolder

                # Baixa arquivos.
                await self._download_folder_contents(
                    remote_id,
                    local_dir,
                    image_extensions,
                )

                paths[f"{split}_{subfolder}"] = local_dir

        return paths

    async def download_validation_from_drive(
        self
    ) -> tuple[Path, Path]:
        """Baixa dados de validação."""

        base = Path(settings.local_data_dir)

        val_labels = base / settings.val_labels_subdir
        val_rgb = base / settings.val_rgb_subdir

        # Resolve IDs remotos.
        val_lbl_id = await self._resolve_subfolder_id(
            settings.base_folder_id,
            "validacao",
            "labels"
        )

        val_rgb_id = await self._resolve_subfolder_id(
            settings.base_folder_id,
            "validacao",
            "rgb"
        )

        img_exts = [
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
        ]

        # Download das máscaras.
        await self._download_folder_contents(
            val_lbl_id,
            val_labels,
            [".png"]
        )

        # Download das imagens RGB.
        await self._download_folder_contents(
            val_rgb_id,
            val_rgb,
            img_exts,
        )

        logger.info(
            "Validation data ready — labels: {} | rgb: {}",
            val_labels,
            val_rgb,
        )

        return val_labels, val_rgb

    # ==========================================================
    # Criação de DataLoaders locais
    # ==========================================================

    async def create_dataloaders(
        self,
        train_rgb_dir: str | Path,
        train_labels_dir: str | Path,
        val_rgb_dir: str | Path,
        val_labels_dir: str | Path,
    ) -> tuple[DataLoader[tuple], DataLoader[tuple]]:
        """Cria DataLoaders locais."""

        # Dataset de treino.
        train_dataset = SegmentationDataset(
            rgb_dir=train_rgb_dir,
            labels_dir=train_labels_dir,
            augmentations=create_train_transforms(),
            image_only_transforms=create_image_only_transforms(),
            preload=settings.preload_dataset,
            copy_paste=settings.aug_copy_paste,
            copy_paste_prob=settings.aug_copy_paste_prob,
            max_ignore_pixel_pct=settings.max_ignore_pixel_pct,
            anthill_duplicate=settings.aug_anthill_duplicate,
            anthill_duplicate_prob=settings.aug_anthill_duplicate_prob,
            anthill_duplicate_max_copies=settings.aug_anthill_duplicate_max_copies,
        )

        # Dataset de validação.
        val_dataset = SegmentationDataset(
            rgb_dir=val_rgb_dir,
            labels_dir=val_labels_dir,
            augmentations=None,
            preload=settings.preload_dataset,
        )

        # Validação dos datasets.
        if len(train_dataset) == 0:
            raise DatasetNotFoundError(
                "No training image-mask pairs found"
            )

        if len(val_dataset) == 0:
            raise DatasetNotFoundError(
                "No validation image-mask pairs found"
            )

        logger.info(
            "Datasets ready — {} train / {} val pairs{}",
            len(train_dataset),
            len(val_dataset),
            " (preloaded in RAM)"
            if settings.preload_dataset
            else "",
        )

        # Configurações do DataLoader.
        use_pin_memory = torch.cuda.is_available()

        num_workers = settings.num_workers

        persistent = num_workers > 0

        prefetch = 2 if num_workers > 0 else None

        # DataLoader de treino.
        train_loader: DataLoader[tuple] = DataLoader(
            train_dataset,
            batch_size=settings.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            collate_fn=_collate_with_names,
        )

        # DataLoader de validação.
        val_loader: DataLoader[tuple] = DataLoader(
            val_dataset,
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            collate_fn=_collate_with_names,
        )

        return train_loader, val_loader

    # DataLoaders locais usando settings

    async def create_local_dataloaders(
        self
    ) -> tuple[DataLoader[tuple], DataLoader[tuple]]:
        """Cria DataLoaders locais usando configurações do projeto."""

        base = Path(settings.local_data_dir)

        train_rgb = base / settings.train_rgb_subdir
        train_labels = base / settings.train_labels_subdir

        val_rgb = base / settings.val_rgb_subdir
        val_labels = base / settings.val_labels_subdir

        # Valida diretórios obrigatórios.
        for path in (
            train_rgb,
            train_labels,
            val_rgb,
            val_labels,
        ):
            if not path.exists():
                raise DatasetNotFoundError(
                    f"Required local directory not found: {path}"
                )

        return await self.create_dataloaders(
            train_rgb_dir=train_rgb,
            train_labels_dir=train_labels,
            val_rgb_dir=val_rgb,
            val_labels_dir=val_labels,
        )

    
    # Streaming datasets
    

    @staticmethod
    def _match_remote_pairs(
        rgb_files: list[dict[str, str]],
        label_files: list[dict[str, str]],
    ) -> list[tuple[dict[str, str], dict[str, str]]]:
        """Relaciona imagens RGB e máscaras remotas."""

        from pathlib import Path as _Path

        pairs: list[
            tuple[dict[str, str], dict[str, str]]
        ] = []

        for rgb in sorted(
            rgb_files,
            key=lambda f: f["name"]
        ):

            # Prefixo usado para match.
            prefix = "_".join(
                _Path(rgb["name"]).stem.split("_")[:4]
            )

            matched = [
                lf
                for lf in label_files
                if _Path(lf["name"]).stem.startswith(prefix)
            ]

            if matched:
                pairs.append((rgb, matched[0]))

        return pairs