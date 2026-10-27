import math

from collision_monitor.logic import LEVEL_CRITICAL, LEVEL_INFO, LEVEL_WARN
from collision_monitor.stability import gravity_specific_force
from collision_monitor.stability_monitor import StabilityMonitor

DIAMOND = [(0.17, 0.0), (0.0, 0.17), (-0.17, 0.0), (0.0, -0.17)]
H = 0.49
# Over a diagonal wheel-to-caster edge. Tipping straight sideways crosses that edge
# at 45 degrees, so the side-slope tip angle is larger: atan(tan(TIP) / cos 45).
TIP = math.atan(0.17 / math.sqrt(2.0) / H)
SIDE_TIP = math.atan(math.tan(TIP) * math.sqrt(2.0))


def make():
    rad = math.radians
    return StabilityMonitor(DIAMOND, (0.0, 0.0, H), rad(15), rad(10), rad(5), rad(5),
                            time_constant=0.5, lowpass_hz=5.0)


def settle(mon, roll, pitch=0.0, steps=300):
    f = gravity_specific_force(roll, pitch)
    for _ in range(steps):
        mon.update_imu((0.0, 0.0, 0.0), f, 0.01)


def test_flat_margin_is_the_static_tip_angle():
    mon = make()
    settle(mon, 0.0)
    assert abs(mon.margin - TIP) < 1e-3
    assert mon.level() == LEVEL_INFO


def test_slope_levels():
    mon = make()
    settle(mon, math.radians(8.0))
    assert mon.level() == LEVEL_WARN
    settle(mon, SIDE_TIP - math.radians(3.0))
    assert mon.level() == LEVEL_CRITICAL
    assert mon.load_transfer > 0.5
    assert abs(mon.margin - math.radians(3.0) * math.cos(math.pi / 4)) < math.radians(0.5)


def test_mount_offset_is_removed():
    mon = make()
    mon.calibrate(math.radians(1.0), math.radians(-0.6))
    # A sensor tilted by the mount offset on level ground.
    settle(mon, math.radians(1.0), math.radians(-0.6))
    assert abs(mon.roll) < 1e-3 and abs(mon.pitch) < 1e-3
    assert abs(mon.margin - TIP) < 1e-3


def test_fast_turn_on_a_slope_is_limited():
    mon = make()
    settle(mon, math.radians(6.0))
    # Already moving at 1 m/s, so the command adds no forward acceleration.
    v, w, k = mon.limit(1.0, -2.0, speed=1.0)
    assert k == 1.0  # turning towards downhill is safe
    v, w, k = mon.limit(1.0, 2.0, speed=1.0)
    assert 0.0 < k < 1.0
    assert mon.predicted_margin(v, w, speed=1.0) >= math.radians(5.0) - 1e-6
