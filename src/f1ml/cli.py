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
def components(
    only: Optional[str] = typer.Option(None, help="Comma list: tyre,pitloss,overtake,flags,start,pace"),
    report: str = typer.Option("reports/components.md", help="Markdown report path (relative to project root)"),
    verbose: bool = typer.Option(False, "-v"),
):
    """Fit the phase-3 component models on all stored races and write features/comp_* tables."""
    from pathlib import Path

    from f1ml.components import COMPONENTS, fit_components
    from f1ml.components.report import benchmark_table, write_components_report

    cfg, store = _setup(verbose)
    chosen = tuple(only.split(",")) if only else COMPONENTS
    tables = fit_components(store, only=chosen)
    for name, df in tables.items():
        store.write_table(df, "features", f"comp_{name}")
    typer.echo(f"wrote {len(tables)} tables: " + ", ".join(f"comp_{n}" for n in tables))
    if set(chosen) == set(COMPONENTS):
        path = Path(report) if Path(report).is_absolute() else cfg.root / report
        write_components_report(tables, path)
        typer.echo(benchmark_table(tables).round(3).to_string())
        typer.echo(f"report: {path}")


def _parse_overrides(items: list[str] | None) -> dict[str, str]:
    out = {}
    for it in items or []:
        drv, _, plan = it.partition("=")
        if not plan:
            raise typer.BadParameter(f"--strategy {it!r}: use DRIVER=PLAN, e.g. VER=S-M-H@14,38")
        out[drv.strip().upper()] = plan.strip()
    return out


@app.command()
def simulate(
    season: int = typer.Option(...),
    round_: int = typer.Option(..., "--round"),
    strategy: Optional[list[str]] = typer.Option(None, help="DRIVER=PLAN, e.g. VER=S-M-H@14,38 (repeatable)"),
    sims: int = typer.Option(5000),
    seed: int = typer.Option(0),
    verbose: bool = typer.Option(False, "-v"),
):
    """Simulate a race from its pre-race data (qualifying must be ingested)."""
    from f1ml.features import build_features
    from f1ml.sim import default_strategies, parse_plan, plan_text, simulate as run
    from f1ml.sim.params import PreraceContext, prerace_params
    from f1ml.sim.replay import components_as_of
    from f1ml.sim.strategy import candidate_plans

    _, store = _setup(verbose)
    feats = build_features(store, include_upcoming=True)
    g = feats[(feats["season"] == season) & (feats["round"] == round_)]
    if g.empty:
        typer.echo(f"no qualifying data for {season} R{round_}")
        raise typer.Exit(1)
    comps = components_as_of(store, g["date"].iloc[0])
    p = prerace_params(store, season, round_, PreraceContext(comps, g))
    strats = default_strategies(p)
    for drv, plan in _parse_overrides(strategy).items():
        strats[p.drivers.index(drv)] = [parse_plan(plan, p.compounds)]
    typer.echo(f"{season} R{round_}: {p.laps} laps, ref lap {p.ref_lap_s:.1f} s, green pit loss "
               f"{p.pit_loss_s[0]:.1f} s, deg S/M/H {p.deg_s.round(3).tolist()} s/lap")
    typer.echo("best default plans: " + ", ".join(f"{plan_text(s, p.compounds)} (+{t - candidate_plans(p)[0][0]:.1f}s)"
                                                  for t, s in candidate_plans(p)[:5]))
    res = run(p, strats, n_sims=sims, seed=seed)
    typer.echo(res.summary().round(3).to_string(index=False))


@app.command()
def whatif(
    season: int = typer.Option(...),
    round_: int = typer.Option(..., "--round"),
    strategy: list[str] = typer.Option(..., help="DRIVER=PLAN for the alternative (repeatable)"),
    sims: int = typer.Option(2000),
    verbose: bool = typer.Option(False, "-v"),
):
    """Replay a finished race as it happened, then with changed strategies (Buscombe's 'What If?')."""
    import numpy as np

    from f1ml.sim import parse_plan, plan_text, simulate as run
    from f1ml.sim.params import replay_inputs
    from f1ml.sim.replay import components_as_of

    _, store = _setup(verbose)
    date = store.query(f"SELECT event_date FROM raw_events WHERE season={season} AND round={round_}")["event_date"].iloc[0]
    ri = replay_inputs(store, season, round_, components_as_of(store, date))
    alt = list(ri.strategies)
    changes = _parse_overrides(strategy)
    for drv, plan in changes.items():
        alt[ri.params.drivers.index(drv)] = parse_plan(plan, ri.params.compounds)
    base = run(ri.params, ri.strategies, n_sims=sims, seed=1, overrides=ri.overrides, stop_jitter=0)
    new = run(ri.params, alt, n_sims=sims, seed=1, overrides=ri.overrides, stop_jitter=0)
    for drv in changes:
        k = ri.params.drivers.index(drv)
        act = ri.actual.set_index("driver").loc[drv]
        typer.echo(f"{drv}: actual P{int(act['finish_order'])} on {plan_text(ri.strategies[k], ri.params.compounds)}")
        for name, r in (("as run", base), ("what if", new)):
            pos = r.position[:, k]
            plan = plan_text(ri.strategies[k] if name == "as run" else alt[k], ri.params.compounds)
            typer.echo(f"  {name:8s} {plan:<14s} median P{np.median(pos):.0f}  P(win) {np.mean(pos == 1):.2f}  "
                       f"P(podium) {np.mean(pos <= 3):.2f}  mean pos {pos.mean():.2f}")


@app.command()
def calibrate(
    from_: int = typer.Option(2024, "--from"),
    to: int = typer.Option(2026, "--to"),
    sims: int = typer.Option(1000),
    report: str = typer.Option("reports/simulator.md"),
    verbose: bool = typer.Option(False, "-v"),
):
    """Replay every dry race with the strategies actually run and compare with the real result."""
    from pathlib import Path

    from f1ml.sim.replay import calibrate as run_cal
    from f1ml.sim.report import write_sim_report

    cfg, store = _setup(verbose)
    per_race, per_driver = run_cal(store, list(range(from_, to + 1)), n_sims=sims,
                                   progress=(lambda m: typer.echo(f"  {m}")) if verbose else None)
    store.write_table(per_race, "features", "sim_calibration")
    store.write_table(per_driver, "features", "sim_calibration_drivers")
    path = Path(report) if Path(report).is_absolute() else cfg.root / report
    write_sim_report(per_race, per_driver, path)
    dry = per_race[~per_race.get("wet", False).fillna(True).astype(bool)] if "wet" in per_race else per_race
    typer.echo(dry[["spearman", "pos_in_10_90", "gap_mae_s", "gap_in_10_90", "winner_correct"]].mean().round(3).to_string())
    typer.echo(f"report: {path}")


@app.command()
def query(sql: str, verbose: bool = typer.Option(False, "-v")):
    """Run SQL over the stored tables (views: raw_laps, clean_laps, features_fit_params, ...)."""
    _, store = _setup(verbose)
    typer.echo(store.query(sql).to_string(index=False))


if __name__ == "__main__":
    app()
