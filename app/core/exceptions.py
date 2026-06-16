"""Domain-specific exception hierarchy.

Services raise these exceptions; the interface layer (FastAPI) catches
them and converts each into the appropriate ``HTTPException``.
"""

from __future__ import annotations


class DomainException(Exception):
    """Base exception for all domain / business-logic errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


#  Model 

class ModelNotLoadedError(DomainException):
    """Raised when attempting to use a model that has not been loaded."""


class TrainingAlreadyInProgressError(DomainException):
    """Raised when a training job is started while another is running."""


#  Data 

class DatasetNotFoundError(DomainException):
    """Raised when expected dataset files cannot be located."""


#  Google Drive 

class GoogleDriveError(DomainException):
    """Raised when a Google Drive operation fails."""


class FolderNotFoundError(GoogleDriveError):
    """Raised when a requested Google Drive folder does not exist."""
