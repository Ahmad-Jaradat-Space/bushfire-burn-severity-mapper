import numpy as np

from src.evaluation.area_estimation import olofsson_area_accuracy


def test_olofsson_closed_form_two_class():
    # cm[i, j] = reference i mapped as j.
    cm = np.array([[90, 10], [5, 95]], dtype=np.float64)
    res = olofsson_area_accuracy(cm, mapped_area_px=None, pixel_area_m2=100.0)

    # Wall-to-wall: area-adjusted overall == naive overall accuracy.
    assert abs(res["overall_accuracy"] - 0.925) < 1e-9

    users = [c["users_accuracy"] for c in res["per_class"]]
    producers = [c["producers_accuracy"] for c in res["per_class"]]
    assert abs(users[0] - 90 / 95) < 1e-9
    assert abs(users[1] - 95 / 105) < 1e-9
    assert abs(producers[0] - 0.9) < 1e-9
    assert abs(producers[1] - 0.95) < 1e-9

    # Estimated reference areas: 0.5 / 0.5 of 200 px * 100 m² = 1 ha each.
    areas = [c["area_adjusted_ha"] for c in res["per_class"]]
    assert abs(areas[0] - 1.0) < 1e-9
    assert abs(areas[1] - 1.0) < 1e-9


def test_olofsson_perfect_map_has_zero_se():
    cm = np.diag([100, 50, 25, 10]).astype(np.float64)
    res = olofsson_area_accuracy(cm)
    assert abs(res["overall_accuracy"] - 1.0) < 1e-12
    assert res["overall_accuracy_se"] < 1e-12
    for c in res["per_class"]:
        assert abs(c["users_accuracy"] - 1.0) < 1e-12
        assert abs(c["producers_accuracy"] - 1.0) < 1e-12


def test_olofsson_ci_contains_point_and_area_adjusts():
    # Map over-predicts class 1: many ref-0 pixels mapped as 1.
    cm = np.array([[60, 40], [5, 95]], dtype=np.float64)
    res = olofsson_area_accuracy(cm, pixel_area_m2=100.0)
    assert res["overall_accuracy_ci"][0] <= res["overall_accuracy"] <= res["overall_accuracy_ci"][1]
    for c in res["per_class"]:
        lo, hi = c["area_ci_ha"]
        assert lo <= c["area_adjusted_ha"] <= hi
    # Mapped class-1 area (135 px) overstates the area-adjusted reference area.
    cls1 = res["per_class"][1]
    assert cls1["area_adjusted_ha"] < cls1["area_mapped_ha"]
