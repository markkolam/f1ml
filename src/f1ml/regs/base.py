"""Regulation plugins.

A regulation module encapsulates the physics/sporting mechanics that differ
between rule sets (DRS vs active aero + 2026 energy, etc.). The race simulator
(phase 4) calls these hooks; the core never branches on the year.

Phase 1 defines the interface, the registry and each module's *data needs*
(which telemetry/timing fields it requires), so ingestion can collect the right
columns per era. Hook bodies are filled in in phase 6.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

_REGISTRY: dict[str, type["RegulationModule"]] = {}


def register(name: str):
    def deco(cls: type[RegulationModule]) -> type[RegulationModule]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"regulation plugin {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def registered_plugins() -> dict[str, type["RegulationModule"]]:
    return dict(_REGISTRY)


def build_plugins(names: list[str], technical: Any) -> list["RegulationModule"]:
    return [_REGISTRY[n](technical) for n in names]


class RegulationModule(ABC):
    name: ClassVar[str] = "base"
    #: extra per-lap columns this module needs from ingestion (optional columns in the contract)
    required_lap_features: ClassVar[tuple[str, ...]] = ()

    def __init__(self, technical: Any):
        self.technical = technical

    # ---- hooks used by the simulator (phase 4/6). Defaults are neutral. ----
    def lap_time_adjust(self, state: Any) -> float:
        """Additive lap-time effect (s) for a car in `state` (e.g. energy clipping)."""
        return 0.0

    def overtake_modifier(self, attacker: Any, defender: Any, track: Any) -> float:
        """Shift (s/lap) applied to the pace delta before the pass S-curve."""
        return 0.0

    def energy_state(self, state: Any) -> Any:
        """Advance per-car energy state by a lap (no-op for eras without an energy model)."""
        return None
