"""Async Google Drive client.

Every blocking API call is dispatched via ``asyncio.to_thread`` so the
event loop is never blocked.  Implements
:class:`~app.domain.protocols.StorageClientProtocol`.
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from loguru import logger  # pyright: ignore[reportMissingImports]

from app.core.config import settings

_MAX_DOWNLOAD_RETRIES: int = 3


class GoogleDriveClient:
    """Async-friendly Google Drive file operations."""

    def __init__(self, credentials_path: str | None = None) -> None:
        self._credentials_path = credentials_path or settings.google_credentials_path
        self._service: Any | None = None

    #  Internals 

    def _authenticate(self) -> Any:
        creds = service_account.Credentials.from_service_account_file(
            self._credentials_path,
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        return build("drive", "v3", credentials=creds)

    def __getstate__(self) -> dict:
        """Drop non-picklable service for DataLoader worker processes."""
        state = self.__dict__.copy()
        state["_service"] = None
        return state

    @property
    def service(self) -> Any:
        if self._service is None:
            self._service = self._authenticate()
        return self._service

    #  Sync primitives 

    def _sync_get_folder_id(
        self, folder_name: str, parent_id: str | None = None,
    ) -> str | None:
        query = (
            f"name='{folder_name}' "
            f"and mimeType='application/vnd.google-apps.folder'"
        )
        if parent_id:
            query += f" and '{parent_id}' in parents"
        result = (
            self.service.files()
            .list(q=query, fields="files(id, name)")
            .execute()
        )
        items: list[dict[str, str]] = result.get("files", [])
        return items[0]["id"] if items else None

    def _sync_list_files(
        self, folder_id: str, extensions: list[str] | None = None,
    ) -> list[dict[str, str]]:
        query = f"'{folder_id}' in parents and trashed=false"
        items: list[dict[str, str]] = []
        page_token: str | None = None

        while True:
            result = (
                self.service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name)",
                    pageSize=1000,
                    pageToken=page_token,
                )
                .execute()
            )
            items.extend(result.get("files", []))
            page_token = result.get("nextPageToken")
            if not page_token:
                break

        if extensions:
            items = [
                f for f in items
                if any(f["name"].lower().endswith(ext) for ext in extensions)
            ]
        return items

    def _sync_download_file(
        self, file_id: str, destination_path: str | None = None,
    ) -> io.BytesIO | str:
        last_exc: Exception | None = None

        for attempt in range(_MAX_DOWNLOAD_RETRIES):
            try:
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

            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_DOWNLOAD_RETRIES - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Download attempt {}/{} failed ({}), retrying in {}s…",
                        attempt + 1, _MAX_DOWNLOAD_RETRIES, exc, wait,
                    )
                    self._service = None
                    time.sleep(wait)

        raise RuntimeError(
            f"Download of '{file_id}' failed after {_MAX_DOWNLOAD_RETRIES} attempts"
        ) from last_exc

    #  Public async API 

    async def get_folder_id(
        self, folder_name: str, parent_id: str | None = None,
    ) -> str | None:
        logger.debug("Looking up folder '{}' (parent='{}')", folder_name, parent_id)
        result = await asyncio.to_thread(
            self._sync_get_folder_id, folder_name, parent_id,
        )
        if result is None:
            logger.warning("Folder '{}' not found", folder_name)
        return result

    async def list_files(
        self, folder_id: str, extensions: list[str] | None = None,
    ) -> list[dict[str, str]]:
        logger.debug("Listing files in folder '{}'", folder_id)
        return await asyncio.to_thread(
            self._sync_list_files, folder_id, extensions,
        )

    async def download_file(
        self, file_id: str, destination_path: str | None = None,
    ) -> io.BytesIO | str:
        logger.debug("Downloading file '{}'", file_id)
        return await asyncio.to_thread(
            self._sync_download_file, file_id, destination_path,
        )
