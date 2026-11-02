import pytest
import yaml

from f1ml.config import ProjectConfig, write_season_file
from f1ml.ingest.base import session_code


def test_real_config_loads_and_rules(project):
    cfg = ProjectConfig(project)
    assert cfg.seasons_in_scope() == list(range(2020, 2027))
    r25 = cfg.rules(2025)
    assert r25.era_id == "2022_ground_effect"
    assert r25.sporting.points.fastest_lap_bonus == 0       # season override applied
    assert cfg.rules(2024).sporting.points.fastest_lap_bonus == 1
    r26 = cfg.rules(2026)
    assert r26.sporting.grid_size == 22 and r26.technical.drs is False and r26.technical.active_aero
    assert r26.plugins == ["active_aero_energy"]
    assert cfg.era_chain("2026_active_aero") == ["2026_active_aero", "2022_ground_effect", "2017_wide_cars"]


def test_track_resolution_and_versions(project):
    cfg = ProjectConfig(project)
    assert cfg.tracks.resolve("Singapore", 2022) == ("marina_bay", "marina_bay@base", True)
    assert cfg.tracks.resolve("Marina Bay", 2024) == ("marina_bay", "marina_bay@2023", True)
    assert cfg.tracks.resolve("Mexico City", 2026)[1] == "rodriguez@2026"
    tid, ver, known = cfg.tracks.resolve("Nowhere City", 2026)
    assert (tid, ver, known) == ("nowhere_city", "nowhere_city@base", False)


def test_new_season_needs_only_a_yaml(project):
    cfg = ProjectConfig(project)
    with pytest.raises(KeyError):
        cfg.rules(2027)
    path = write_season_file(cfg, 2027)
    cfg2 = ProjectConfig(project)
    assert path.exists() and cfg2.rules(2027).era_id == "2026_active_aero"


def test_new_era_with_new_plugin_is_config_plus_module(project):
    """A future rule reset: close the current era, add a new era YAML with a new plugin."""
    from f1ml.regs import RegulationModule, register, registered_plugins

    if "test_future_aero" not in registered_plugins():
        @register("test_future_aero")
        class FutureAero(RegulationModule):
            required_lap_features = ("speed_st",)

    era26 = project / "config" / "eras" / "2026_active_aero.yaml"
    d = yaml.safe_load(era26.read_text())
    d["last_season"] = 2029
    era26.write_text(yaml.safe_dump(d))
    new = dict(d, id="2030_future", name="future", first_season=2030, last_season=None,
               transfer_from="2026_active_aero", plugins=["test_future_aero"])
    (project / "config" / "eras" / "2030_future.yaml").write_text(yaml.safe_dump(new))
    (project / "config" / "seasons" / "2030.yaml").write_text("season: 2030\nera: 2030_future\n")
    cfg = ProjectConfig(project)
    assert cfg.rules(2030).plugins == ["test_future_aero"]
    assert cfg.era_chain("2030_future")[:2] == ["2030_future", "2026_active_aero"]


@pytest.mark.parametrize("mutation, message", [
    (("seasons/2024.yaml", "season: 2024\nera: nope\n"), "unknown era"),
    (("seasons/2024.yaml", "season: 2024\nera: 2026_active_aero\n"), "outside era"),
])
def test_validation_errors(project, mutation, message):
    rel, text = mutation
    (project / "config" / rel).write_text(text)
    with pytest.raises(ValueError, match=message):
        ProjectConfig(project)


def test_unknown_plugin_rejected(project):
    p = project / "config" / "eras" / "2022_ground_effect.yaml"
    d = yaml.safe_load(p.read_text())
    d["plugins"] = ["warp_drive"]
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match="warp_drive"):
        ProjectConfig(project)


def test_overlapping_eras_rejected(project):
    p = project / "config" / "eras" / "2022_ground_effect.yaml"
    d = yaml.safe_load(p.read_text())
    d["last_season"] = 2026
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match="overlap"):
        ProjectConfig(project)


def test_pirelli_nominations(project):
    (project / "config" / "pirelli" / "2024.yaml").write_text(
        "season: 2024\nnominations:\n  13: {hard: c3, medium: c4, soft: c5}\n")
    cfg = ProjectConfig(project)
    assert cfg.compound_map(2024, 13) == {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"}
    assert cfg.compound_map(2024, 1) == {}


@pytest.mark.parametrize("name, fmt, code", [
    ("Practice 1", "conventional", "FP1"), ("Race", "conventional", "R"),
    ("Sprint Qualifying", "sprint", "S"),            # 2021: the sprint race
    ("Sprint Qualifying", "sprint_qualifying", "SQ"),  # 2024+: the shootout
    ("Sprint Shootout", "sprint_shootout", "SQ"), ("Sprint", "sprint_shootout", "S"),
    ("Testing", "testing", None),
])
def test_session_codes(name, fmt, code):
    assert session_code(name, fmt) == code
