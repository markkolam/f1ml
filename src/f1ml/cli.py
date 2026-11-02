"""f1ml command line.

    f1ml update --season 2026 --round 21        # one race weekend
    f1ml update --season 2026 --round latest    # most recent completed weekend
    f1ml update --season 2026                   # whole season (skips what is up to date)
    f1ml backfill --from 2020 --to 2026         # everything in scope
    f1ml season-init 2027 [--era 2026_active_aero]
    f1ml validate-config | tracks-check --season 2026 | status | query "SELECT ..."
"""

from __future__ import annotations

import logging
from typing import Optional

import typer

from f1ml.config import ProjectConfig, write_season_file
from f1ml.store import Store

app = typer.Typer(add_completion=False, no_args_is_help=True, help="F1 race & strategy modelling stack")


def _setup(verbose: bool) -> tuple[ProjectConfig, Store]:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = ProjectConfig()
    return cfg, Store(cfg.data_dir)


def livetiming_reachable(timeout: float = 10.0) -> bool:
    import requests

    try:
        requests.head("https://livetiming.formula1.com/static/2024/Index.json", timeout=timeout)
        return True
    except requests.RequestException:
        return False


def _pipeline(cfg: ProjectConfig, store: Store, with_pit_stops: bool, source: str = "auto"):
    from f1ml.ingest.jolpica import JolpicaClient
    from f1ml.pipeline import Pipeline

    src = cfg.settings.sources
    jolpica = JolpicaClient(src.jolpica_base_url, src.jolpica_min_interval_s)
    if source == "auto":
        source = "fastf1" if livetiming_reachable() else "openf1"
        typer.echo(f"source: {source}" + ("" if source == "fastf1" else
                                          " (livetiming.formula1.com unreachable; OpenF1 covers 2023+)"))
    if source == "fastf1":
        from f1ml.ingest.fastf1_source import FastF1Source

        t = cfg.settings.telemetry
        data_source = FastF1Source(cfg.data_dir / "cache" / "fastf1", (t.window_low_kph, t.window_high_kph))
    elif source == "openf1":
        from f1ml.ingest.openf1_source import OpenF1Source

        data_source = OpenF1Source(cfg.data_dir / "cache", jolpica)
    else:
        raise typer.BadParameter(f"unknown source {source!r} (auto|fastf1|openf1)")
    return Pipeline(cfg, store, data_source, jolpica if with_pit_stops else None)


def _print_reports(reports) -> bool:
    ok = True
    for r in reports:
        typer.echo(f"{r.season} R{r.round:02d} {r.event_name}")
        for step, st in r.steps.items():
            typer.echo(f"    {step:<14} {st}")
        ok &= r.ok
    if not reports:
        typer.echo("nothing to do (no completed events matched)")
    return ok


@app.command()
def update(
    season: int = typer.Option(..., help="Season to update"),
    round_: Optional[str] = typer.Option(None, "--round", help="Round number, comma list, or 'latest'"),
    sessions: Optional[str] = typer.Option(None, help="Comma list of session codes (FP1,FP2,FP3,SQ,S,Q,R)"),
    force: bool = typer.Option(False, help="Re-ingest and re-clean even if up to date"),
    telemetry: bool = typer.Option(False, help="Also compute per-lap telemetry features (slow)"),
    pit_stops: bool = typer.Option(True, help="Fetch pit-stop records from Jolpica"),
    source: str = typer.Option("auto", help="auto | fastf1 | openf1"),
    reclean: bool = typer.Option(False, help="Re-run cleaning/fits on stored raw data (no re-download)"),
    verbose: bool = typer.Option(False, "-v"),
):
    """Ingest + clean + fit one season (or selected rounds). Safe to re-run."""
    cfg, store = _setup(verbose)
    pipe = _pipeline(cfg, store, pit_stops, source)
    rounds = None
    if round_ == "latest":
        done = pipe.completed_rounds(season)
        if not done:
            typer.echo(f"no completed rounds in {season} yet")
            raise typer.Exit(0)
        rounds = [max(done)]
    elif round_:
        rounds = [int(x) for x in round_.split(",")]
    sess = [s.strip().upper() for s in sessions.split(",")] if sessions else None
    reports = pipe.update(season, rounds, force=force, telemetry=telemetry, sessions=sess, reclean=reclean)
    raise typer.Exit(0 if _print_reports(reports) else 1)


@app.command()
def backfill(
    from_: int = typer.Option(None, "--from", help="First season (default: min_season)"),
    to: int = typer.Option(None, "--to", help="Last season (default: latest configured)"),
    telemetry: bool = typer.Option(False),
    pit_stops: bool = typer.Option(True),
    source: str = typer.Option("auto", help="auto | fastf1 | openf1"),
    reclean: bool = typer.Option(False, help="Re-run cleaning/fits on stored raw data (no re-download)"),
    verbose: bool = typer.Option(False, "-v"),
):
    """Run `update` for every configured season in range."""
    from f1ml.ingest.openf1_source import OPENF1_FIRST_SEASON

    cfg, store = _setup(verbose)
    pipe = _pipeline(cfg, store, pit_stops, source)
    seasons = [s for s in cfg.seasons_in_scope() if (from_ is None or s >= from_) and (to is None or s <= to)]
    ok = True
    for s in seasons:
        typer.echo(f"=== {s}")
        if pipe.source.name == "openf1" and s < OPENF1_FIRST_SEASON:
            typer.echo(f"    skipped: OpenF1 has no data before {OPENF1_FIRST_SEASON}; needs FastF1 (livetiming)")
            ok = False
            continue
        try:
            ok &= _print_reports(pipe.update(s, telemetry=telemetry, reclean=reclean))
        except Exception as exc:  # noqa: BLE001 - one bad season must not stop the backfill
            typer.echo(f"    error: {type(exc).__name__}: {exc}")
            ok = False
    raise typer.Exit(0 if ok else 1)


@app.command("season-init")
def season_init(season: int, era: Optional[str] = typer.Option(None, help="Era id (default: the era covering the season)")):
    """Create config/seasons/<season>.yaml for a new season."""
    cfg = ProjectConfig()
    path = write_season_file(cfg, season, era)
    ProjectConfig(cfg.root)  # re-validate with the new file
    typer.echo(f"created {path}")


@app.command("validate-config")
def validate_config():
    """Load and cross-check every config file."""
    cfg = ProjectConfig()
    typer.echo(f"ok: {len(cfg.eras)} eras, {len(cfg.seasons)} seasons, {len(cfg.tracks.tracks)} tracks")
    for s in cfg.seasons_in_scope():
        r = cfg.rules(s)
        typer.echo(f"  {s}: era={r.era_id} grid={r.sporting.grid_size} drs={r.technical.drs} "
                   f"active_aero={r.technical.active_aero} plugins={r.plugins} "
                   f"fl_bonus={r.sporting.points.fastest_lap_bonus}")


@app.command("tracks-check")
def tracks_check(season: int, verbose: bool = typer.Option(False, "-v")):
    """List the season's venues and whether each is in the track registry."""
    cfg, store = _setup(verbose)
    pipe = _pipeline(cfg, store, False, "auto")
    missing = 0
    for ev in pipe.source.schedule(season):
        tid, ver, known = cfg.tracks.resolve(ev.circuit_id or ev.location, season)
        missing += not known
        typer.echo(f"R{ev.round:02d} {ev.location:<22} -> {ver:<24} {'ok' if known else 'NOT IN REGISTRY'}")
    raise typer.Exit(1 if missing else 0)


@app.command()
def status(verbose: bool = typer.Option(False, "-v")):
    """What has been ingested, per round."""
    from f1ml.pipeline import status_table

    _, store = _setup(verbose)
    df = status_table(store)
    typer.echo(df.to_string(index=False) if len(df) else "no data ingested yet")


@app.command()
def backtest(
    from_: int = typer.Option(2024, "--from", help="First season to evaluate"),
    to: int = typer.Option(2026, "--to", help="Last season to evaluate"),
    models: str = typer.Option("grid,quali,lgbm", help="Comma list from the model zoo"),
    min_train_races: int = typer.Option(10),
    sims: int = typer.Option(5000, help="Monte Carlo draws per race for podium/top-10 probabilities"),
    report: str = typer.Option("reports/backtest.md", help="Markdown report path (relative to project root)"),
    verbose: bool = typer.Option(False, "-v"),
):
    """Walk-forward backtest: each race predicted by a model trained only on earlier races."""
    from pathlib import Path

    from f1ml.eval import walk_forward
    from f1ml.eval.report import write_report
    from f1ml.features import FEATURES, build_features
    from f1ml.models.zoo import MODEL_ZOO

    cfg, store = _setup(verbose)
    feats = build_features(store, include_upcoming=False)
    store.write_table(feats, "features", "prerace")
    chosen = {m: MODEL_ZOO[m] for m in models.split(",")}
    res = walk_forward(feats, chosen, list(range(from_, to + 1)), min_train_races=min_train_races, n_sims=sims,
                       progress=(lambda m: typer.echo(f"  {m}")) if verbose else None)
    store.write_table(res.per_race, "features", "backtest_per_race")
    store.write_table(res.predictions, "features", "backtest_predictions")
    imp = None
    if "lgbm" in chosen:
        imp = MODEL_ZOO["lgbm"]().fit(feats.dropna(subset=["finish_order"])).importance()
    path = Path(report) if Path(report).is_absolute() else cfg.root / report
    write_report(res, path, imp, notes=f"Features: {', '.join(FEATURES)}.")
    typer.echo(res.summary().round(3).to_string())
    typer.echo(f"report: {path}")


@app.command()
def predict(
    season: int = typer.Option(...),
    round_: int = typer.Option(..., "--round"),
    model: str = typer.Option("lgbm"),
    sims: int = typer.Option(20000),
    verbose: bool = typer.Option(False, "-v"),
):
    """Predict a race from its practice + qualifying data (run `update` for the weekend first)."""
    from f1ml.features import build_features
    from f1ml.models.zoo import MODEL_ZOO

    _, store = _setup(verbose)
    feats = build_features(store, include_upcoming=True)
    target = feats[(feats["season"] == season) & (feats["round"] == round_)]
    if target.empty:
        typer.echo(f"no qualifying data for {season} R{round_}; run `f1ml update --season {season} --round {round_}`")
        raise typer.Exit(1)
    train = feats[(feats["date"] < target["date"].iloc[0]) & feats["finish_order"].notna()]
    m = MODEL_ZOO[model]().fit(train)
    p = m.predict(target, n_sims=sims)
    out = target[["driver", "team", "grid_pos"]].join(p).sort_values("exp_pos")
    if target["finish_order"].notna().any():
        out["actual"] = target["finish_order"]
    typer.echo(f"{season} R{round_}: trained on {train[['season', 'round']].drop_duplicates().shape[0]} races, "
               f"tau={m.tau:.2f}")
    typer.echo(out.round(3).to_string(index=False))


@app.command()
def query(sql: str, verbose: bool = typer.Option(False, "-v")):
    """Run SQL over the stored tables (views: raw_laps, clean_laps, features_fit_params, ...)."""
    _, store = _setup(verbose)
    typer.echo(store.query(sql).to_string(index=False))


if __name__ == "__main__":
    app()
