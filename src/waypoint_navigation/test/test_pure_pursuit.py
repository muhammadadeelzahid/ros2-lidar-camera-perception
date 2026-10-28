import math

from waypoint_navigation.pure_pursuit import (
    advance,
    command,
    lookahead_point,
    remaining_distance,
    stopping_speed,
    to_vehicle,
    turn_command,
)

L_PATH = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)]


def test_to_vehicle_frame():
    x, y = to_vehicle((1.0, 1.0, math.pi / 2), (1.0, 3.0))
    assert math.isclose(x, 2.0, abs_tol=1e-9) and math.isclose(y, 0.0, abs_tol=1e-9)


def test_lookahead_on_a_straight_segment():
    assert lookahead_point(L_PATH, 0, (1.0, 0.2), 1.0) == (2.0, 0.0)


def test_lookahead_wraps_round_a_corner():
    x, y = lookahead_point(L_PATH, 0, (3.5, 0.0), 1.0)
    assert math.isclose(x, 4.0) and math.isclose(y, 0.5)


def test_lookahead_stops_at_the_end():
    assert lookahead_point(L_PATH, 1, (4.0, 3.8), 1.0) == (4.0, 4.0)


def test_advance_moves_past_finished_segments():
    assert advance(L_PATH, 0, (3.0, 0.0)) == 0
    assert advance(L_PATH, 0, (4.0, 0.1)) == 1
    assert advance(L_PATH, 1, (4.0, 3.0)) == 1  # the last segment is never left


def test_remaining_distance():
    assert math.isclose(remaining_distance(L_PATH, 0, (1.0, 0.0)), 7.0)
    assert math.isclose(remaining_distance(L_PATH, 1, (4.0, 1.0)), 3.0)


def test_straight_ahead_drives_straight():
    v, w = command((0.0, 0.0, 0.0), (1.0, 0.0), 0.5, 1.0)
    assert v == 0.5 and w == 0.0


def test_target_to_the_left_turns_left_on_an_arc():
    v, w = command((0.0, 0.0, 0.0), (1.0, 0.5), 0.5, 2.0)
    # curvature 2 * 0.5 / 1.25 = 0.8
    assert v == 0.5 and math.isclose(w, 0.4)


def test_tight_arc_slows_down_to_respect_max_yaw_rate():
    v, w = command((0.0, 0.0, 0.0), (0.6, 0.6), 1.0, 0.5)
    assert math.isclose(abs(w), 0.5) and v < 1.0


def test_target_behind_turns_on_the_spot():
    v, w = command((0.0, 0.0, 0.0), (-1.0, 0.2), 0.5, 1.0)
    assert v == 0.0 and w == 1.0


def test_stopping_speed():
    assert stopping_speed(10.0, 0.5, 0.5) == 0.5
    assert math.isclose(stopping_speed(0.04, 0.5, 0.5), 0.2)
    assert stopping_speed(0.0, 0.5, 0.5) == 0.05


def test_turn_command_takes_the_short_way():
    w, err = turn_command(math.radians(170), math.radians(-170), 1.0)
    assert w > 0.0 and math.isclose(err, math.radians(20))
