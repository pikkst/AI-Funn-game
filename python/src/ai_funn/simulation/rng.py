from __future__ import annotations

import random
from typing import Any


class WorldRNG:
    def __init__(self, seed: int | None = None, state: Any = None) -> None:
        if state is not None:
            self._rng = random.Random()
            self._rng.setstate(state)
        else:
            self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice_index(self, n: int) -> int:
        return self._rng.randrange(n) if n > 0 else 0

    def get_state(self) -> Any:
        return self._rng.getstate()

    def set_state(self, state: Any) -> None:
        self._rng.setstate(state)

    @classmethod
    def from_state(cls, state: Any) -> WorldRNG:
        return cls(state=state)
