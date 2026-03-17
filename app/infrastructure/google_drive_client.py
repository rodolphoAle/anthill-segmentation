"""Async wrapper around the synchronous *google-api-python-client*.

Every blocking Google API call is dispatched to the default thread-pool
via ``asyncio.to_thread`` so the event loop is never blocked.

Implements :class:`~app.domain.protocols.StorageClientProtocol`.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from loguru import logger # pyright: ignore[reportMissingImports]

from app.core.config import settings


class GoogleDriveClient:
    """Async-friendly Google Drive file operations.

    Args:
        credentials_path: Path to the service-account JSON key file.
    """

    def __init__(self, credentials_path: str | None = None) -> None:
        self._credentials_path: str = (
            credentials_path or settings.google_credentials_path
        )
        self._service: Any | None = None

    #  private helpers (sync) 

    def _authenticate(self) -> Any:
        scopes = ["https://www.googleapis.com/auth/drive"]
        credentials = service_account.Credentials.from_service_account_file(
            self._credentials_path, scopes=scopes,
        )
        return build("drive", "v3", credentials=credentials)

    @property
    def service(self) -> Any:
        """Lazily authenticated Drive service object."""
        if self._service is None:
            self._service = self._authenticate()
        return self._service

    def _sync_get_folder_id(
        self, folder_name: str, parent_id: str | None = None,
    ) -> str | None:
        query = (
            f"name='{folder_name}' "
            f"and mimeType='application/vnd.google-apps.folder'"
        )
        if parent_id:
            query += f" and '{parent_id}' in parents"
        results = (
            self.service.files()
            .list(q=query, fields="files(id, name)")
            .execute()
        )
        items: list[dict[str, str]] = results.get("files", [])
        return items[0]["id"] if items else None

    def _sync_list_files(
        self, folder_id: str, extensions: list[str] | None = None,
    ) -> list[dict[str, str]]:
        query = f"'{folder_id}' in parents and trashed=false"
        results = (
            self.service.files()
            .list(q=query, fields="files(id, name)")
            .execute()
        )
        items: list[dict[str, str]] = results.get("files", [])
        if extensions:
            items = [
                item
                for item in items
                if any(item["name"].lower().endswith(ext) for ext in extensions)
            ]
        return items

    def _sync_download_file(
        self, file_id: str, destination_path: str | None = None,
    ) -> io.BytesIO | str:
        request = self.service.files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        buffer.seek(0)

        if destination_path:
            with open(destination_path, "wb") as fh:
                fh.write(buffer.read())
            return destination_path
        return buffer

    #  public async API 

    async def get_folder_id(
        self, folder_name: str, parent_id: str | None = None,
    ) -> str | None:
        """Look up a folder by name inside an optional parent."""
        logger.debug(
            "Looking up folder '{}' under parent '{}'",
            folder_name,
            parent_id,
        )
        result = await asyncio.to_thread(
            self._sync_get_folder_id, folder_name, parent_id,
        )
        if result is None:
            logger.warning("Folder '{}' not found", folder_name)
        return result

    async def list_files(
        self, folder_id: str, extensions: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """List files in a Drive folder, optionally filtered by extension."""
        logger.debug("Listing files in folder '{}'", folder_id)
        return await asyncio.to_thread(
            self._sync_list_files, folder_id, extensions,
        )

    async def download_file(
        self, file_id: str, destination_path: str | None = None,
    ) -> io.BytesIO | str:
        """Download a single file from Google Drive."""
        logger.debug("Downloading file '{}'", file_id)
        return await asyncio.to_thread(
            self._sync_download_file, file_id, destination_path,
        )
