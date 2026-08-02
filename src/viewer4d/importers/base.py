from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

from viewer4d.core.model import Viewer4DGS


class BaseImporter(ABC):
    """Base interface for source-project importers."""

    @abstractmethod
    def convert(
        self,
        source: str | Path,
        **kwargs: Any,
    ) -> Viewer4DGS | Mapping[str, Any]:
        """Convert a source file to a portable viewer model or payload."""
        raise NotImplementedError
