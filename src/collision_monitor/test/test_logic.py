from collision_monitor.logic import (
    LEVEL_CRITICAL,
    LEVEL_NONE,
    assess,
    circumscribed_radius,
)

EGO_R = circumscribed_radius(0.4, 0.3)
OBJ_R = circumscribed_radius(0.6, 0.6)


def test_approaching_object_is_critical():
    # Surface ~0.8 m ahead, closing at 1 m/s -> ttc ~0.8 s.
    rel = (0.8 + EGO_R + OBJ_R, 0.0)
    res = assess(rel, (-1.0, 0.0), EGO_R, OBJ_R)
    assert res['level'] == LEVEL_CRITICAL
    assert res['closing_speed'] > 0.0
    assert res['ttc'] < 1.0


def test_far_object_is_none():
    res = assess((10.0, 0.0), (0.0, 0.0), EGO_R, OBJ_R)
    assert res['level'] == LEVEL_NONE


def test_receding_object_has_infinite_ttc():
    res = assess((2.0, 0.0), (1.0, 0.0), EGO_R, OBJ_R)
    assert res['ttc'] == float('inf')
