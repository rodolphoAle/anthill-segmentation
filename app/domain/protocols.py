"""Contratos usados pela camada de serviço.

Os serviços dependem deste protocolo, e não de uma implementação
concreta como Google Drive ou armazenamento local.
"""

from __future__ import annotations

import io
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageClientProtocol(Protocol):
    "Contrato para operações assíncronas de armazenamento."

    async def get_folder_id(
        self,
        folder_name: str,
        parent_id: str | None = None,
    ) -> str | None:
        """Busca o ID de uma pasta pelo nome."""
        ...

    async def list_files(
        self,
        folder_id: str,
        extensions: list[str] | None = None,
    ) -> list[dict[str, str]]:
        "Lista arquivos de uma pasta, com filtro opcional por extensão."
        ...

    async def download_file(
        self,
        file_id: str,
        destination_path: str | None = None,
    ) -> io.BytesIO | str:
        "Baixa um arquivo para memória ou para o disco."
        ...