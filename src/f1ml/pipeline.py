"""The `f1ml update` pipeline: ingest -> validate -> clean/tag -> fit -> features.

Idempotent: each (round, step) is recorded in a manifest and skipped on re-run
unless `force=True`, the schema version changed, or an upstream step re-ran.
A failure in one session is recorded and the rest of the round continues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from f1ml.clean import apply_corrections, extract_long_runs, fit_race, tag_laps
from f1ml.config import ProjectConfig
from f1ml.ingest.base import PitStopSource, ScheduledEvent, Source
from f1ml.store import Store

log = logging.getLogger(__name__)

RACE_CODES = ("R", "S")
# Timing data is usually complete a little after the chequered flag.
AVAILABILITY_DELAY = timedelta(hours=3)


@dataclass
class RoundReport:
    season: int
    round: int
    event_name: str
    steps: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(v.startswith("error") for v in self.steps.values())


class Pipeline:
    def __init__(self, cfg: ProjectConfig, store: Store, source: Source,
                 pit_source: PitStopSource | None = None, now: pd.Timestamp | None = None):
        self.cfg = cfg
        self.store = store
        self.source = source
        self.pit_source = pit_source
        self.now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)

    # ------------------------------------------------------------------ public
    def update(self, season: int, rounds: list[int] | None = None, *, force: bool = False,
               telemetry: bool = False, sessions: list[str] | None = None,
               reclean: bool = False) -> list[RoundReport]:
        if season < self.cfg.settings.min_season:
            raise ValueError(f"season {season} is before min_season {self.cfg.settings.min_season}")
        self.cfg.rules(season)  # fails early with a helpful message if the season isn't configured
        events = self.source.schedule(season)
        if rounds:
            events = [e for e in events if e.round in set(rounds)]
        reports = []
        for ev in events:
            if ev.event_date is not None and pd.Timestamp(ev.event_date) - timedelta(days=4) > self.now:
                continue  # whole weekend is in the future
            reports.append(self._update_round(ev, force=force, telemetry=telemetry, only=sessions,
                                              reclean=reclean))
        return reports

    def completed_rounds(self, season: int) -> list[int]:
        out = []
        for ev in self.source.schedule(season):
            race = [ts for code, _, ts in ev.session_codes() if code == "R"]
            if race and race[0] is not None and self._available(race[0]):
                out.append(ev.round)
        return out

    # ------------------------------------------------------------------ internals
    def _available(self, ts: pd.Timestamp | None) -> bool:
        if ts is None:
            return True
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts + AVAILABILITY_DELAY <= self.now

    def _update_round(self, ev: ScheduledEvent, *, force: bool, telemetry: bool,
                      only: list[str] | None, reclean: bool = False) -> RoundReport:
        season, rnd = ev.season, ev.round
        rep = RoundReport(season, rnd, ev.event_name)
        man = self.store.load_manifest(season, rnd)
        man["event_name"] = ev.event_name

        track_id, track_version, known = self.cfg.tracks.resolve(ev.circuit_id or ev.location, season)
        if not known and ev.circuit_id:
            alt = self.cfg.tracks.resolve(ev.location, season)
            if alt[2]:
                track_id, track_version, known = alt
        era_id = self.cfg.rules(season).era_id
        codes = ev.session_codes()
        events_df = pd.DataFrame([{
            "season": season, "round": rnd, "event_name": ev.event_name, "location": ev.location,
            "country": ev.country, "event_format": ev.event_format, "event_date": ev.event_date,
            "track_id": track_id, "track_version": track_version, "track_known": known,
            "era_id": era_id, "sessions": ",".join(c for c, _, _ in codes),
        }])
        self.store.write(events_df, "raw", "events", season, rnd)
        if not known:
            rep.steps["track"] = f"warning: location {ev.location!r} not in tracks registry (id={track_id})"

        compound_map = self.cfg.compound_map(season, rnd)
        for code, name, ts in codes:
            if only and code not in only:
                continue
            if not self._available(ts):
                rep.steps[f"ingest:{code}"] = "not_yet_available"
                continue
            ingested_now = self._ingest(ev, code, name, man, rep, force=force, telemetry=telemetry)
            if Store.step_done(man, f"ingest:{code}"):
                self._clean(ev, code, man, rep, force=force or ingested_now or reclean,
                            track=(track_id, track_version), era_id=era_id, compound_map=compound_map)

        if self.pit_source is not None and Store.step_done(man, "ingest:R") and (only is None or "R" in only):
            self._pit_stops(ev, man, rep, force=force)

        self.store.save_manifest(man)
        return rep

    def _ingest(self, ev, code, name, man, rep, *, force, telemetry) -> bool:
        step = f"ingest:{code}"
        need_tel = telemetry and not self.store.exists("features", "lap_window", ev.season, ev.round, code)
        if Store.step_done(man, step) and not force and not need_tel:
            rep.steps[step] = "skipped (up to date)"
            return False
        try:
            data = self.source.load_session(ev, name, code, telemetry=telemetry)
            keys = {"season": ev.season, "round": ev.round, "session": code}
            for table, df in (("laps", data.laps), ("results", data.results),
                              ("weather", data.weather), ("race_control", data.race_control)):
                self.store.write(df.assign(**keys), "raw", table, ev.season, ev.round, code)
            if data.lap_window is not None:
                self.store.write(data.lap_window.assign(**keys), "features", "lap_window", ev.season, ev.round, code)
            Store.mark(man, step, "ok", laps=len(data.laps), total_laps=data.total_laps, source=self.source.name)
            rep.steps[step] = f"ok ({len(data.laps)} laps)"
            return True
        except Exception as exc:  # noqa: BLE001 - recorded, round continues
            log.debug("ingest failed %s %s %s", ev.season, ev.round, code, exc_info=True)
            Store.mark(man, step, "error", error=f"{type(exc).__name__}: {exc}")
            rep.steps[step] = f"error: {type(exc).__name__}: {exc}"
            return False

    def _clean(self, ev, code, man, rep, *, force, track, era_id, compound_map) -> None:
        step = f"clean:{code}"
        if Store.step_done(man, step) and not force:
            rep.steps[step] = "skipped (up to date)"
            return
        try:
            s = self.cfg.settings
            laps = self.store.read("raw", "laps", ev.season, ev.round, code)
            weather = self.store.read("raw", "weather", ev.season, ev.round, code)
            tagged = tag_laps(laps, weather, dirty_air_gap_s=s.cleaning.dirty_air_gap_s,
                              outlier_pct=s.cleaning.outlier_pct, compound_map=compound_map)
            tagged["track_id"], tagged["track_version"] = track
            tagged["era_id"] = era_id

            fit = None
            info: dict = {}
            if code in RACE_CODES:
                n = s.normalisation
                fit = fit_race(tagged, fuel_prior=(n.fuel_effect_prior_mean, n.fuel_effect_prior_sd),
                               deg_prior=(n.deg_prior_mean, n.deg_prior_sd), huber_k=n.huber_k,
                               min_laps=n.min_laps_for_fit)
                total = man["steps"][f"ingest:{code}"].get("total_laps")
                tagged = apply_corrections(tagged, fit, total)
                if fit is not None:
                    keys = {"season": ev.season, "round": ev.round, "session": code}
                    self.store.write(fit.params.assign(**keys), "features", "fit_params", ev.season, ev.round, code)
                    info = {"fuel_effect": round(fit.fuel_effect, 4), "sigma": round(fit.sigma, 3),
                            "fit_laps": fit.n_laps, "deg": {k: round(v, 4) for k, v in fit.deg.items()}}
                else:
                    info = {"fit": "insufficient clean laps"}
            else:
                tagged = apply_corrections(tagged, None, None)

            self.store.write(tagged, "clean", "laps", ev.season, ev.round, code, contract="clean_laps")
            lr = extract_long_runs(tagged, min_laps=s.cleaning.min_long_run_laps,
                                   fuel_effect_prior=s.normalisation.fuel_effect_prior_mean)
            keys = {"season": ev.season, "round": ev.round, "session": code}
            self.store.write(lr.assign(**keys), "features", "long_runs", ev.season, ev.round, code)
            info.update(clean_laps=int(tagged["is_clean"].sum()), long_runs=len(lr),
                        c_number_missing=not bool(compound_map))
            Store.mark(man, step, "ok", **info)
            rep.steps[step] = "ok " + ", ".join(f"{k}={v}" for k, v in info.items() if k != "deg")
        except Exception as exc:  # noqa: BLE001
            log.debug("clean failed %s %s %s", ev.season, ev.round, code, exc_info=True)
            Store.mark(man, step, "error", error=f"{type(exc).__name__}: {exc}")
            rep.steps[step] = f"error: {type(exc).__name__}: {exc}"

    def _pit_stops(self, ev, man, rep, *, force) -> None:
        step = "pit_stops"
        if Store.step_done(man, step) and not force:
            rep.steps[step] = "skipped (up to date)"
            return
        try:
            df = self.pit_source.pit_stops(ev.season, ev.round)
            self.store.write(df.assign(season=ev.season, round=ev.round), "raw", "pit_stops", ev.season, ev.round)
            Store.mark(man, step, "ok", stops=len(df))
            rep.steps[step] = f"ok ({len(df)} stops)"
        except Exception as exc:  # noqa: BLE001
            log.debug("pit stops failed %s %s", ev.season, ev.round, exc_info=True)
            Store.mark(man, step, "error", error=f"{type(exc).__name__}: {exc}")
            rep.steps[step] = f"error: {type(exc).__name__}: {exc}"


def status_table(store: Store) -> pd.DataFrame:
    rows = []
    for m in store.manifests():
        steps = m.get("steps", {})
        rows.append({
            "season": m["season"], "round": m["round"], "event": m.get("event_name", ""),
            "ok": sum(1 for s in steps.values() if s.get("status") == "ok"),
            "errors": sum(1 for s in steps.values() if s.get("status") == "error"),
            "sessions": ",".join(sorted(k.split(":")[1] for k, v in steps.items()
                                        if k.startswith("clean:") and v.get("status") == "ok")),
            "c_numbers": "missing" if any(v.get("c_number_missing") for v in steps.values()) else "ok",
            "source": ",".join(sorted({v["source"] for v in steps.values() if v.get("source")})),
            "updated_at": m.get("updated_at", ""),
        })
    return pd.DataFrame(rows)
