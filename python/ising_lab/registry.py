"""Best-known-energy registry for benchmark instances.

The registry maps a per-instance key to the lowest energy ever observed,
along with the state that achieved it and a note about where it came from
(brute_force, sa, pt, neal, dwave, ...). Updates are monotone in energy:
only strictly better energies replace an existing entry, so once a
brute-force optimum is in, sampler results can't push it back up.

The registry persists to JSON, so you can grow it across runs:

    reg = OptimumRegistry("sk_optima.json")
    benchmark(samplers, instances, registry=reg)
    reg.save()                          # now persisted

Use cases:
    - At N <= 28 we brute-force the truth; at larger N we lean on the best
      sampler result so far and tighten it as we learn more.
    - Comparing samplers: the registry's `source` field shows which sampler
      currently holds the best for each instance.
    - Sharing benchmark suites: a registry file is a self-contained record
      of "the best we know" plus the configurations that achieved them.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

PathLike = Union[str, Path]


@dataclass
class BestKnown:
    """A single best-known solution record."""

    energy: float
    state: List[int]
    source: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OptimumRegistry:
    """A monotone-in-energy registry of best-known solutions.

    Updates only succeed if the new energy is strictly lower than the
    previous best (within `tol`). Equal-energy results from a different
    sampler do not displace the original finder.
    """

    def __init__(self, path: Optional[PathLike] = None, tol: float = 1e-12) -> None:
        self._records: Dict[str, BestKnown] = {}
        self._path: Optional[Path] = Path(path) if path is not None else None
        self._tol = tol
        if self._path is not None and self._path.exists():
            self.load()

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def best(self, key: str) -> Optional[BestKnown]:
        """Return the current best record for `key`, or None if unknown."""
        return self._records.get(key)

    def update(
        self,
        key: str,
        energy: float,
        state: List[int],
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Insert/replace if `energy` strictly beats the previous best.

        Returns True if the registry was updated, False otherwise.
        """
        current = self._records.get(key)
        if current is not None and energy >= current.energy - self._tol:
            return False
        self._records[key] = BestKnown(
            energy=float(energy),
            state=[int(s) for s in state],
            source=source,
            timestamp=time.time(),
            metadata=dict(metadata) if metadata else {},
        )
        return True

    def remove(self, key: str) -> Optional[BestKnown]:
        return self._records.pop(key, None)

    def save(self, path: Optional[PathLike] = None) -> Path:
        """Persist the registry to JSON."""
        target = Path(path) if path is not None else self._path
        if target is None:
            raise ValueError("registry has no path configured; pass one to save()")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: asdict(v) for k, v in self._records.items()}
        target.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return target

    def load(self, path: Optional[PathLike] = None) -> None:
        """Replace in-memory records with those from `path` (or self.path)."""
        target = Path(path) if path is not None else self._path
        if target is None:
            raise ValueError("registry has no path configured; pass one to load()")
        data = json.loads(target.read_text())
        self._records = {k: BestKnown(**v) for k, v in data.items()}

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, key: object) -> bool:
        return key in self._records

    def __iter__(self) -> Iterator[str]:
        return iter(self._records)

    def items(self) -> List[Tuple[str, BestKnown]]:
        return list(self._records.items())


def sk_instance_key(instance) -> str:
    """Canonical key for an SKInstance: 'sk-<dist>-n<N>-seed<seed>'."""
    return f"sk-{instance.distribution}-n{instance.n}-seed{instance.seed}"


__all__ = ["BestKnown", "OptimumRegistry", "sk_instance_key"]
