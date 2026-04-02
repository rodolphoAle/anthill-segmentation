"""Async service for dataset preparation and DataLoader creation.

Responsibilities
----------------
* Download images from a remote storage backend (Google Drive, etc.)
  to the local filesystem.
* Create PyTorch ``DataLoader`` instances for training and validation.

The service depends on :class:`~app.domain.protocols.StorageClientProtocol`
so it is decoupled from any concrete storage implementation
(**Dependency Inversion**).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from loguru import logger
from torch.utils.data import DataLoader

from app.core.config import settings
from app.core.exceptions import DatasetNotFoundError, FolderNotFoundError
from app.domain.protocols import StorageClientProtocol
from app.infrastructure.segmentation_dataset import SegmentationDataset
from app.infrastructure.streaming_dataset import StreamingSegmentationDataset


def _collate_with_names(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Custom collate that stacks tensors and keeps filenames as a list."""
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    names = [item[2] for item in batch]
    return images, masks, names


def create_train_transforms() -> transforms.Compose:
    """Build geometric + photometric augmentations from settings.

    All transforms that affect spatial layout (flip, rotation) are applied
    jointly to image AND mask.  ColorJitter is image-only and is appended
    outside the joint pipeline — the caller in SegmentationDataset.__getitem__
    applies self._augmentations to both, so ColorJitter must NOT be included
    here when joint application would corrupt the mask.

    Note: torchvision v2 transforms handle (image, mask) pairs correctly for
    geometric ops.  ColorJitter is safe on PIL images and is only applied to
    the image tensor AFTER the joint step via a second transform in the dataset.

    Returns an empty Compose if all augmentations are disabled.
    """
    transform_list: list[transforms.Transform] = []

    if settings.aug_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if settings.aug_vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip())

    if settings.aug_rotation_degrees > 0:
        transform_list.append(transforms.RandomRotation(settings.aug_rotation_degrees))

    return transforms.Compose(transform_list)


def create_image_only_transforms() -> transforms.Compose | None:
    """Photometric augmentations applied to the image tensor only (not the mask).

    Returns None if color jitter is disabled, so callers can skip the step.
    """
    if not settings.aug_color_jitter:
        return None
    return transforms.Compose([
        transforms.ColorJitter(
            brightness=settings.aug_color_jitter_brightness,
            contrast=settings.aug_color_jitter_contrast,
            saturation=settings.aug_color_jitter_saturation,
        )
    ])


#  Service 

class DataService:
    """Stateless async service for dataset operations.

    Args:
        storage_client: An object satisfying
            :class:`~app.domain.protocols.StorageClientProtocol`.
            Required when ``data_mode == "online"``.
    """

    def __init__(
        self,
        storage_client: StorageClientProtocol | None = None,
    ) -> None:
        self._storage_client = storage_client

    #  remote download 

    async def _resolve_subfolder_id(
        self,
        parent_id: str,
        *names: str,
    ) -> str:
        """Walk a chain of folder names and return the final folder id."""
        if self._storage_client is None:
            raise DatasetNotFoundError(
                "No storage client configured for online mode"
            )

        current_id = parent_id
        for name in names:
            folder_id = await self._storage_client.get_folder_id(
                name, current_id,
            )
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
        """Download every file in *folder_id* to *local_dir*.

        Files that already exist locally are skipped.

        Returns:
            Number of newly downloaded files.
        """
        if self._storage_client is None:
            raise DatasetNotFoundError(
                "No storage client configured for online mode"
            )

        local_dir.mkdir(parents=True, exist_ok=True)
        files = await self._storage_client.list_files(
            folder_id, extensions=extensions,
        )

        downloaded = 0
        for file_info in files:
            dest = local_dir / file_info["name"]
            if dest.exists():
                continue
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
        """Download train + validation splits from Google Drive.

        Returns:
            Mapping of ``"<split>_<type>"`` → local directory path.
        """
        destination = Path(destination)
        image_extensions = [".png", ".jpg", ".jpeg", ".tif"]
        paths: dict[str, Path] = {}

        for split in ("treino", "validacao"):
            for subfolder in ("rgb", "labels"):
                remote_id = await self._resolve_subfolder_id(
                    base_folder_id, split, subfolder,
                )
                local_dir = destination / split / subfolder
                await self._download_folder_contents(
                    remote_id, local_dir, image_extensions,
                )
                paths[f"{split}_{subfolder}"] = local_dir

        return paths

    #  DataLoader creation 

    async def create_dataloaders(
        self,
        train_rgb_dir: str | Path,
        train_labels_dir: str | Path,
        val_rgb_dir: str | Path,
        val_labels_dir: str | Path,
    ) -> tuple[DataLoader[tuple], DataLoader[tuple]]:
        """Build training and validation ``DataLoader`` instances.

        Raises:
            DatasetNotFoundError: If either dataset has zero image-mask
                pairs.
        """
        train_dataset = SegmentationDataset(
            rgb_dir=train_rgb_dir,
            labels_dir=train_labels_dir,
            augmentations=create_train_transforms(),
            image_only_transforms=create_image_only_transforms(),
            preload=settings.preload_dataset,
        )
        val_dataset = SegmentationDataset(
            rgb_dir=val_rgb_dir,
            labels_dir=val_labels_dir,
            augmentations=None,
            preload=settings.preload_dataset,
        )

        if len(train_dataset) == 0:
            raise DatasetNotFoundError("No training image-mask pairs found")
        if len(val_dataset) == 0:
            raise DatasetNotFoundError("No validation image-mask pairs found")

        logger.info(
            "Datasets ready — {} train / {} val pairs{}",
            len(train_dataset),
            len(val_dataset),
            " (preloaded in RAM)" if settings.preload_dataset else "",
        )

        use_pin_memory = torch.cuda.is_available()
        num_workers = settings.num_workers
        persistent = num_workers > 0
        prefetch = 2 if num_workers > 0 else None

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

    #  Offline (local disk) DataLoaders 

    async def create_local_dataloaders(self) -> tuple[DataLoader[tuple], DataLoader[tuple]]:
        """Build DataLoaders from local disk using paths configured in settings.

        Expected structure under ``settings.local_data_dir``::

            <local_data_dir>/
              <train_rgb_subdir>/       ← e.g. training/rgb/rgb
              <train_labels_subdir>/    ← e.g. training/labels/labels
              <val_rgb_subdir>/         ← e.g. validation/rgb/rgb
              <val_labels_subdir>/      ← e.g. validation/labels/labels

        Raises:
            DatasetNotFoundError: If a required directory does not exist or
                either split has no matched pairs.
        """
        base = Path(settings.local_data_dir)
        train_rgb = base / settings.train_rgb_subdir
        train_labels = base / settings.train_labels_subdir
        val_rgb = base / settings.val_rgb_subdir
        val_labels = base / settings.val_labels_subdir

        for path in (train_rgb, train_labels, val_rgb, val_labels):
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

    #  Streaming DataLoaders (no disk writes) 

    @staticmethod
    def _match_remote_pairs(
        rgb_files: list[dict[str, str]],
        label_files: list[dict[str, str]],
    ) -> list[tuple[dict[str, str], dict[str, str]]]:
        """Match remote RGB files to their label counterparts by name prefix."""
        from pathlib import Path as _Path

        pairs: list[tuple[dict[str, str], dict[str, str]]] = []
        for rgb in sorted(rgb_files, key=lambda f: f["name"]):
            prefix = "_".join(_Path(rgb["name"]).stem.split("_")[:4])
            matched = [
                lf
                for lf in label_files
                if _Path(lf["name"]).stem.startswith(prefix)
            ]
            if matched:
                pairs.append((rgb, matched[0]))
        return pairs

    async def create_streaming_dataloaders_from_drive(
        self,
        base_folder_id: str,
    ) -> tuple[DataLoader[tuple], DataLoader[tuple]]:
        """Build streaming DataLoaders backed directly by Google Drive.

        File metadata (IDs + names) are fetched upfront; actual image
        bytes are downloaded on-demand inside each ``__getitem__`` call
        and released immediately — **nothing is written to disk**.

        The Drive service object inside ``GoogleDriveClient`` is lazily
        initialised, so each DataLoader worker re-authenticates
        automatically on first use.  Use ``num_workers=0`` (the default
        here) to avoid any worker-process pickling concerns.

        Args:
            base_folder_id: Root Drive folder that contains
                ``treino/rgb``, ``treino/labels``,
                ``validacao/rgb``, and ``validacao/labels`` sub-folders.

        Returns:
            ``(train_loader, val_loader)`` tuple.

        Raises:
            DatasetNotFoundError: If storage client is not configured or
                either split has no matched pairs.
        """
        if self._storage_client is None:
            raise DatasetNotFoundError(
                "No storage client configured for online mode"
            )

        img_exts = [".png", ".jpg", ".jpeg", ".tif"]

        # Resolve all four remote folder IDs in sequence
        train_rgb_id = await self._resolve_subfolder_id(
            base_folder_id, "treino", "rgb"
        )
        train_lbl_id = await self._resolve_subfolder_id(
            base_folder_id, "treino", "labels"
        )
        val_rgb_id = await self._resolve_subfolder_id(
            base_folder_id, "validacao", "rgb"
        )
        val_lbl_id = await self._resolve_subfolder_id(
            base_folder_id, "validacao", "labels"
        )

        train_rgb_files = await self._storage_client.list_files(
            train_rgb_id, img_exts
        )
        train_lbl_files = await self._storage_client.list_files(
            train_lbl_id, [".png"]
        )
        val_rgb_files = await self._storage_client.list_files(
            val_rgb_id, img_exts
        )
        val_lbl_files = await self._storage_client.list_files(
            val_lbl_id, [".png"]
        )

        train_pairs = self._match_remote_pairs(train_rgb_files, train_lbl_files)
        val_pairs = self._match_remote_pairs(val_rgb_files, val_lbl_files)

        if not train_pairs:
            raise DatasetNotFoundError(
                "No training image-mask pairs found on Drive"
            )
        if not val_pairs:
            raise DatasetNotFoundError(
                "No validation image-mask pairs found on Drive"
            )

        logger.info(
            "Streaming datasets ready — {} train / {} val pairs (no disk writes)",
            len(train_pairs),
            len(val_pairs),
        )

        # Pass the synchronous download method so workers never call asyncio
        download_fn = self._storage_client._sync_download_file  # type: ignore[attr-defined]

        train_dataset = StreamingSegmentationDataset(
            pairs=train_pairs,
            download_fn=download_fn,
            augmentations=create_train_transforms(),
        )
        val_dataset = StreamingSegmentationDataset(
            pairs=val_pairs,
            download_fn=download_fn,
            augmentations=None,
        )

        # num_workers > 0: multiple worker processes download images in
        # parallel.  GoogleDriveClient resets its service to None when
        # pickled so each worker re-authenticates lazily on first use.
        num_workers = settings.num_workers
        use_pin_memory = torch.cuda.is_available()
        persistent = num_workers > 0
        prefetch = 2 if num_workers > 0 else None

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
