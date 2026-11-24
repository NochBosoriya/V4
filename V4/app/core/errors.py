"""Application error types with bilingual (English/Khmer) messages."""
from __future__ import annotations


class AppError(Exception):
    """Base error carrying an English and a Khmer message."""

    def __init__(self, en: str, km: str = ""):
        super().__init__(en)
        self.en = en
        self.km = km or en

    def __str__(self) -> str:
        return self.en


class CancelledError(RuntimeError):
    """Raised by long operations when their cancel event is set."""


class EngineNotAvailable(AppError):
    """A required engine/package/model is not installed or not found."""


class ProviderError(AppError):
    """An AI provider call failed."""

    def __init__(self, en: str, km: str = "", provider: str = "",
                 status_code: int | None = None, retryable: bool = False):
        super().__init__(en, km)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
