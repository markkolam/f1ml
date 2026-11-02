"""Parquet storage + DuckDB query layer + per-round manifests.

Layout (one file per partition, written atomically):

    data/<layer>/<table>/season=YYYY/round=RR/session=CODE/part.parquet
    data/<layer>/<table>/season=YYYY/round=RR/part.parquet          (round-level tables)
    data/manifest/YYYY_RR.json

Partition values are also stored as columns inside each file, so files are
self-describing and can be read without hive partitioning.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from f1ml import SCHEMA_VERSION
from f1ml.contracts import conform

LAYERS = ("raw", "clean", "features")


class Store:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir)

    # ------------------------------------------------------------------ paths
    def part_path(self, layer: str, table: str, season: int, round_: int, session: str | None = None) -> Path:
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer}")
        p = self.root / layer / table / f"season={season}" / f"round={round_:02d}"
        if session is not None:
            p = p / f"session={session}"
        return p / "part.parquet"

    # ------------------------------------------------------------------ io
    def write(self, df: pd.DataFrame, layer: str, table: str, season: int, round_: int,
              session: str | None = None, contract: str | None = None) -> Path:
        df = conform(df, contract or table)
        path = self.part_path(layer, table, season, round_, session)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(fd)
        try:
            df.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return path

    def write_table(self, df: pd.DataFrame, layer: str, table: str) -> Path:
        """Unpartitioned derived table (rebuilt wholesale, e.g. feature tables, backtests)."""
        path = self.root / layer / table / "all.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return path

    def read(self, layer: str, table: str, season: int, round_: int, session: str | None = None) -> pd.DataFrame:
        return pd.read_parquet(self.part_path(layer, table, season, round_, session))

    def exists(self, layer: str, table: str, season: int, round_: int, session: str | None = None) -> bool:
        return self.part_path(layer, table, season, round_, session).exists()

    def glob(self, layer: str, table: str) -> str:
        return str(self.root / layer / table / "**" / "*.parquet")

    def has_table(self, layer: str, table: str) -> bool:
        return any((self.root / layer / table).rglob("*.parquet"))

    # ------------------------------------------------------------------ duckdb
    def connect(self) -> duckdb.DuckDBPyConnection:
        """In-memory DuckDB with a view per stored table, named <layer>_<table>."""
        con = duckdb.connect()
        for layer in LAYERS:
            base = self.root / layer
            if not base.exists():
                continue
            for tdir in sorted(p for p in base.iterdir() if p.is_dir()):
                if any(tdir.rglob("*.parquet")):
                    con.execute(
                        f"CREATE VIEW {layer}_{tdir.name} AS SELECT * FROM read_parquet("
                        f"'{tdir.as_posix()}/**/*.parquet', hive_partitioning=false, union_by_name=true)"
                    )
        return con

    def query(self, sql: str) -> pd.DataFrame:
        with self.connect() as con:
            return con.execute(sql).df()

    # ------------------------------------------------------------------ manifests
    def manifest_path(self, season: int, round_: int) -> Path:
        return self.root / "manifest" / f"{season}_{round_:02d}.json"

    def load_manifest(self, season: int, round_: int) -> dict[str, Any]:
        p = self.manifest_path(season, round_)
        if not p.exists():
            return {"season": season, "round": round_, "schema_version": SCHEMA_VERSION, "steps": {}}
        return json.loads(p.read_text())

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        p = self.manifest_path(manifest["season"], manifest["round"])
        p.parent.mkdir(parents=True, exist_ok=True)
        manifest["schema_version"] = SCHEMA_VERSION
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        os.replace(tmp, p)

    @staticmethod
    def step_done(manifest: dict[str, Any], step: str) -> bool:
        st = manifest.get("steps", {}).get(step)
        return bool(st) and st.get("status") == "ok" and manifest.get("schema_version") == SCHEMA_VERSION

    @staticmethod
    def mark(manifest: dict[str, Any], step: str, status: str, **info: Any) -> None:
        manifest.setdefault("steps", {})[step] = {
            "status": status,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **info,
        }

    def manifests(self) -> list[dict[str, Any]]:
        d = self.root / "manifest"
        return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []
