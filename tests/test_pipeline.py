import pandas as pd
import pytest

from f1ml.config import ProjectConfig
from f1ml.contracts import ContractError, conform
from f1ml.pipeline import Pipeline, status_table
from f1ml.store import Store
from tests.conftest import FakePits, FakeSource


def make(project, source=None):
    cfg = ProjectConfig(project)
    store = Store(cfg.data_dir)
    src = source or FakeSource()
    return Pipeline(cfg, store, src, FakePits(), now=pd.Timestamp("2024-08-01")), store, src


def test_end_to_end_and_idempotent(project):
    pipe, store, src = make(project)
    reports = pipe.update(2024)
    assert [r.round for r in reports] == [1]           # future round skipped
    rep = reports[0]
    assert rep.ok, rep.steps
    assert rep.steps["ingest:R"].startswith("ok") and rep.steps["clean:R"].startswith("ok")
    assert rep.steps["pit_stops"].startswith("ok")
    assert store.exists("clean", "laps", 2024, 1, "R")
    assert store.exists("features", "fit_params", 2024, 1, "R")
    assert store.exists("features", "long_runs", 2024, 1, "FP2")
    assert not store.exists("features", "fit_params", 2024, 1, "FP2")  # no fuel fit on practice

    man = store.load_manifest(2024, 1)
    assert man["steps"]["clean:R"]["fuel_effect"] == pytest.approx(0.05, abs=0.01)

    n_calls = len(src.calls)
    rep2 = pipe.update(2024)[0]
    assert len(src.calls) == n_calls                    # nothing re-downloaded
    assert all("skipped" in v for v in rep2.steps.values())

    pipe.update(2024, [1], force=True, sessions=["R"])
    assert len(src.calls) == n_calls + 1


def test_duckdb_views_and_status(project):
    pipe, store, _ = make(project)
    pipe.update(2024)
    q = store.query("""
        SELECT session, count(*) n, sum(is_clean::INT) clean, any_value(track_version) tv
        FROM clean_laps GROUP BY session ORDER BY session""")
    assert list(q.session) == ["FP2", "R"]
    assert (q.tv == "hungaroring@base").all()
    fuel = store.query("SELECT value FROM features_fit_params WHERE param='fuel_effect'")
    assert len(fuel) == 1
    st = status_table(store)
    assert st.loc[0, "errors"] == 0 and st.loc[0, "c_numbers"] == "missing"


def test_unknown_track_warns(project):
    pipe, store, _ = make(project, FakeSource(location="Atlantis"))
    rep = pipe.update(2024)[0]
    assert rep.steps["track"].startswith("warning")
    assert store.read("raw", "events", 2024, 1).track_id[0] == "atlantis"


def test_source_failure_is_recorded_not_fatal(project):
    class Broken(FakeSource):
        def load_session(self, event, session_name, code, telemetry=False):
            if code == "FP2":
                raise ConnectionError("403 from proxy")
            return super().load_session(event, session_name, code, telemetry)

    pipe, store, _ = make(project, Broken())
    rep = pipe.update(2024)[0]
    assert rep.steps["ingest:FP2"].startswith("error") and rep.steps["clean:R"].startswith("ok")
    assert not rep.ok


def test_season_before_scope_rejected(project):
    pipe, _, _ = make(project)
    with pytest.raises(ValueError, match="min_season"):
        pipe.update(2019)


def test_contract_adds_optional_and_keeps_extras():
    df = pd.DataFrame({"season": [2026], "round": [1], "session": ["R"], "driver": ["X"], "driver_number": ["1"],
                       "lap_number": [3], "lap_time_s": [90.1], "time_s": [300.0], "compound": ["SOFT"],
                       "tyre_life": [3.0], "track_status": ["1"], "energy_clip_kph": [4.2]})
    out = conform(df, "laps")
    assert out["speed_st"].isna().all()
    assert out.columns[-1] == "energy_clip_kph"
    with pytest.raises(ContractError):
        conform(df.drop(columns=["compound"]), "laps")
