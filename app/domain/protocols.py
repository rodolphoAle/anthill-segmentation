"""Structural typing contracts (Protocols) consumed by the service layer.

Following the **Interface Segregation** and **Dependency Inversion**
principles, high-level services depend on these thin protocols — never
on concrete infrastructure classes.
"""

from __future__ import annotations

import io
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageClientProtocol(Protocol):
    """Async cloud / local storage operations."""

    async def get_folder_id(
        self,
        folder_name: str,
        parent_id: str | None = None,
    ) -> str | None:
        """Return the folder identifier for *folder_name*, or ``None``."""
        ...

    async def list_files(
        self,
        folder_id: str,
        extensions: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """List files inside *folder_id*, optionally filtered by extension."""
        ...

    async def download_file(
        self,
        file_id: str,
        destination_path: str | None = None,
    ) -> io.BytesIO | str:
        """Download a file by *file_id*.

        When *destination_path* is provided the file is written to disk and
        the path is returned; otherwise an in-memory ``BytesIO`` is returned.
        """
        ...
