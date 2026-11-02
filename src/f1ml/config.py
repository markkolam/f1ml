"""Configuration: settings, regulation eras, seasons, tracks, Pirelli nominations.

Everything season- or rule-specific lives in YAML under ``config/``. Adding a
season means adding ``config/seasons/<year>.yaml``; a rule reset means adding
``config/eras/<id>.yaml`` (plus a regulation plugin if new physics is needed).
No Python changes are required for either.
"""

from __future__ import annotations

import copy
import os
import re
import unicodedata
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def find_project_root(start: Path | None = None) -> Path:
    """Project root = $F1ML_ROOT, else the nearest parent with config/settings.yaml."""
    env = os.environ.get("F1ML_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for p in (here, *here.parents):
        if (p / "config" / "settings.yaml").is_file():
            return p
    return PACKAGE_ROOT


def _load_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# --------------------------------------------------------------------------- settings
class CleaningSettings(BaseModel):
    dirty_air_gap_s: float = 1.5
    outlier_pct: float = 0.05
    min_long_run_laps: int = 5


class NormalisationSettings(BaseModel):
    fuel_effect_prior_mean: float = 0.045
    fuel_effect_prior_sd: float = 0.03
    deg_prior_mean: float = 0.05
    deg_prior_sd: float = 0.15
    huber_k: float = 1.5
    min_laps_for_fit: int = 60


class TelemetrySettings(BaseModel):
    window_low_kph: float = 70.0
    window_high_kph: float = 150.0


class SourceSettings(BaseModel):
    jolpica_base_url: str = "https://api.jolpi.ca/ergast/f1"
    jolpica_min_interval_s: float = 0.3


class Settings(BaseModel):
    data_dir: str = "data"
    min_season: int = 2020
    cleaning: CleaningSettings = Field(default_factory=CleaningSettings)
    normalisation: NormalisationSettings = Field(default_factory=NormalisationSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    sources: SourceSettings = Field(default_factory=SourceSettings)


# --------------------------------------------------------------------------- eras
class Points(BaseModel):
    race: list[int]
    fastest_lap_bonus: int = 0
    sprint: list[int] = Field(default_factory=list)


class Sporting(BaseModel):
    model_config = ConfigDict(extra="allow")
    cars_per_team: int = 2
    grid_size: int
    mandatory_dry_compounds: int = 2
    points: Points


class Technical(BaseModel):
    # extra="allow": a future era can add fields (e.g. new energy limits)
    # without touching this file; plugins read what they need.
    model_config = ConfigDict(extra="allow")
    wheel_rim_in: int
    drs: bool
    active_aero: bool = False
    mgu_h: bool = True
    mguk_max_kw: float | None = None
    refuelling: bool = False
    ground_effect_floor: bool = False


class Era(BaseModel):
    id: str
    name: str
    first_season: int
    last_season: int | None = None
    transfer_from: str | None = None
    technical: Technical
    sporting: Sporting
    plugins: list[str] = Field(default_factory=list)

    def covers(self, season: int) -> bool:
        return season >= self.first_season and (self.last_season is None or season <= self.last_season)


# --------------------------------------------------------------------------- seasons
class SeasonFile(BaseModel):
    season: int
    era: str
    compound_range: list[str] | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)


class SeasonRules(BaseModel):
    """An era with a season's overrides applied: the rules actually in force."""

    season: int
    era_id: str
    compound_range: list[str] | None
    technical: Technical
    sporting: Sporting
    plugins: list[str]


# --------------------------------------------------------------------------- tracks
class TrackVersion(BaseModel):
    from_season: int
    label: str


class Track(BaseModel):
    id: str
    aliases: list[str] = Field(default_factory=list)
    versions: list[TrackVersion] = Field(default_factory=list)

    def version_for(self, season: int) -> str:
        """Version tag in force for a season, e.g. 'marina_bay@2023' or 'monza@base'."""
        applicable = [v.from_season for v in self.versions if v.from_season <= season]
        return f"{self.id}@{max(applicable)}" if applicable else f"{self.id}@base"


class TrackRegistry(BaseModel):
    tracks: list[Track] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> "TrackRegistry":
        ids = [t.id for t in self.tracks]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"duplicate track ids: {sorted(dup)}")
        seen: dict[str, str] = {}
        for t in self.tracks:
            for a in t.aliases:
                key = a.casefold()
                if key in seen and seen[key] != t.id:
                    raise ValueError(f"alias {a!r} used by both {seen[key]} and {t.id}")
                seen[key] = t.id
        return self

    @cached_property
    def _alias_index(self) -> dict[str, Track]:
        idx = {}
        for t in self.tracks:
            idx[t.id.casefold()] = t
            for a in t.aliases:
                idx[a.casefold()] = t
        return idx

    def resolve(self, location: str, season: int) -> tuple[str, str, bool]:
        """Return (track_id, track_version, known)."""
        t = self._alias_index.get(location.casefold())
        if t is None:
            tid = slugify(location)
            return tid, f"{tid}@base", False
        return t.id, t.version_for(season), True


# --------------------------------------------------------------------------- project
class ProjectConfig:
    """Loads and cross-validates every config file under ``<root>/config``."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else find_project_root()
        cfg = self.root / "config"
        self.settings = Settings(**_load_yaml(cfg / "settings.yaml")) if (cfg / "settings.yaml").exists() else Settings()
        self.eras: dict[str, Era] = {}
        for p in sorted((cfg / "eras").glob("*.yaml")):
            era = Era(**_load_yaml(p))
            if era.id in self.eras:
                raise ValueError(f"duplicate era id {era.id} ({p})")
            self.eras[era.id] = era
        self.seasons: dict[int, SeasonFile] = {}
        for p in sorted((cfg / "seasons").glob("*.yaml")):
            s = SeasonFile(**_load_yaml(p))
            if p.stem != str(s.season):
                raise ValueError(f"{p.name}: file name must match season {s.season}")
            self.seasons[s.season] = s
        reg_path = cfg / "tracks" / "registry.yaml"
        self.tracks = TrackRegistry(**_load_yaml(reg_path)) if reg_path.exists() else TrackRegistry()
        self._pirelli: dict[int, dict[int, dict[str, str]]] = {}
        for p in sorted((cfg / "pirelli").glob("*.yaml")):
            raw = _load_yaml(p)
            season = int(raw["season"])
            noms = {int(r): {str(k).upper(): str(v).upper() for k, v in m.items()}
                    for r, m in (raw.get("nominations") or {}).items()}
            self._pirelli[season] = noms
        self.validate()

    # ---------------------------------------------------------------- validation
    def validate(self) -> None:
        from f1ml.regs import registered_plugins  # local import: avoid cycle

        errors: list[str] = []
        known_plugins = set(registered_plugins())
        for era in self.eras.values():
            if era.transfer_from and era.transfer_from not in self.eras:
                errors.append(f"era {era.id}: transfer_from {era.transfer_from!r} is not a known era")
            for pl in era.plugins:
                if pl not in known_plugins:
                    errors.append(f"era {era.id}: plugin {pl!r} not registered (known: {sorted(known_plugins)})")
        eras = sorted(self.eras.values(), key=lambda e: e.first_season)
        for a, b in zip(eras, eras[1:]):
            if a.last_season is None or a.last_season >= b.first_season:
                errors.append(f"eras {a.id} and {b.id} overlap")
        for s in self.seasons.values():
            era = self.eras.get(s.era)
            if era is None:
                errors.append(f"season {s.season}: unknown era {s.era!r}")
            elif not era.covers(s.season):
                errors.append(f"season {s.season}: outside era {era.id} range")
            else:
                try:
                    self.rules(s.season)
                except Exception as exc:  # pydantic error from bad overrides
                    errors.append(f"season {s.season}: overrides invalid: {exc}")
        if errors:
            raise ValueError("config validation failed:\n  - " + "\n  - ".join(errors))

    # ---------------------------------------------------------------- accessors
    @property
    def data_dir(self) -> Path:
        p = Path(self.settings.data_dir)
        return p if p.is_absolute() else self.root / p

    def era_for(self, season: int) -> Era:
        if season in self.seasons:
            return self.eras[self.seasons[season].era]
        for era in self.eras.values():
            if era.covers(season):
                return era
        raise KeyError(f"no era covers season {season}")

    def rules(self, season: int) -> SeasonRules:
        if season not in self.seasons:
            raise KeyError(f"season {season} has no config/seasons/{season}.yaml (run `f1ml season-init {season}`)")
        s = self.seasons[season]
        era = self.eras[s.era]
        merged = deep_merge(era.model_dump(include={"technical", "sporting", "plugins"}), s.overrides)
        return SeasonRules(
            season=season,
            era_id=era.id,
            compound_range=s.compound_range,
            technical=Technical(**merged["technical"]),
            sporting=Sporting(**merged["sporting"]),
            plugins=list(merged.get("plugins", era.plugins)),
        )

    def era_chain(self, era_id: str) -> list[str]:
        """Era and its ancestors via transfer_from, newest first."""
        chain, cur = [], era_id
        while cur and cur not in chain:
            chain.append(cur)
            cur = self.eras[cur].transfer_from
        return chain

    def compound_map(self, season: int, round_: int) -> dict[str, str]:
        """{'SOFT': 'C5', ...} for a round, or {} if nominations are not configured."""
        return self._pirelli.get(season, {}).get(round_, {})

    def seasons_in_scope(self) -> list[int]:
        return sorted(s for s in self.seasons if s >= self.settings.min_season)


def write_season_file(cfg: ProjectConfig, season: int, era_id: str | None = None) -> Path:
    """Create config/seasons/<season>.yaml (used by `f1ml season-init`)."""
    path = cfg.root / "config" / "seasons" / f"{season}.yaml"
    if path.exists():
        raise FileExistsError(path)
    if era_id is None:
        era_id = cfg.era_for(season).id
    if era_id not in cfg.eras:
        raise KeyError(f"unknown era {era_id!r}; add config/eras/{era_id}.yaml first")
    era = cfg.eras[era_id]
    if not era.covers(season):
        raise ValueError(f"era {era_id} does not cover {season} (first={era.first_season}, last={era.last_season})")
    text = (
        "# Calendar, teams and drivers are read from the timing data at ingest time;\n"
        "# this file only carries what the data cannot tell us.\n"
        f"season: {season}\nera: {era_id}\ncompound_range: null\noverrides: {{}}\n"
    )
    path.write_text(text, encoding="utf-8")
    return path
