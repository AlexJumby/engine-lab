from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseLambdaController(ABC):
    def reset(self) -> None:
        return None

    @abstractmethod
    def compute(
        self,
        *,
        t: float,
        state: Dict[str, float],
        a: float,
    ) -> tuple[float, Dict[str, Any]]:
        raise NotImplementedError
