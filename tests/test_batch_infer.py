from src.inference.batch_infer import project_cost


def test_project_cost_scales_inversely_with_throughput():
    slow = project_cost(1_000_000)
    fast = project_cost(2_000_000)
    # twice the throughput -> half the time and half the cost
    assert abs(fast["hours"] - slow["hours"] / 2) < 1e-9
    assert abs(fast["usd"] - slow["usd"] / 2) < 1e-9


def test_project_cost_pixel_count_at_resolution():
    # 80 Mha at 0.5 m = 80e6 * 1e4 m2 / 0.25 m2/px = 3.2e12 px
    p = project_cost(1_000_000, hectares=80_000_000, res_m=0.5)
    assert abs(p["pixels"] - 3.2e12) / 3.2e12 < 1e-9
    assert p["hours"] > 0 and p["usd"] > 0
