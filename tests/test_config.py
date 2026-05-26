from src.utils.config import load_config


def test_root_config_loads():
    cfg = load_config("configs/config.yaml")
    assert cfg.project.name == "australian-bushfire-burn-severity-mapper"
    assert cfg.data.tile_size == 256
    assert cfg.class_map.ignore_id == 255


def test_experiment_extends_root():
    cfg = load_config("configs/experiments/baseline_dnbr.yaml")
    # Inherited from root
    assert cfg.data.tile_size == 256
    assert cfg.events.test == ["east_gippsland_2019_2020"]
    # Defined in experiment
    assert cfg.experiment.name == "baseline_dnbr"
    assert cfg.experiment.event == "kangaroo_island_2019_2020"
    assert cfg.baseline.binary_thresholds == [0.10, 0.20, 0.30]
    # `extends` should be stripped from the merged result
    assert "extends" not in cfg


def test_cli_overrides():
    cfg = load_config(
        "configs/experiments/baseline_dnbr.yaml",
        overrides=["data.tile_size=128", "experiment.name=override_test"],
    )
    assert cfg.data.tile_size == 128
    assert cfg.experiment.name == "override_test"
