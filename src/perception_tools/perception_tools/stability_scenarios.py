#!/usr/bin/env python3
"""Drive the robot through tip-over scenarios in slopes.world, gate off then on.

Each scenario places the robot, accelerates straight, then drives a turn or carries
on straight, sending commands on cmd_vel_in so they pass through the collision
monitor. The stability gate is switched through the monitor's stability_gate
parameter. Ground truth comes from Gazebo: a wheel has lifted once its contact point
is more than lift_height (10 cm) above the surface the robot stands on. Bouncing over
the 2 cm bumps and skid-steer chatter lift a wheel up to about 5 cm. The bump run is
there to count false alarms.

Run with the simulation started from scripts/stability_scenarios.sh. The simulation is
not deterministic, so each scenario is repeated (repeats parameter). Prints a summary
and writes per-sample data and per-run results to output_dir.
"""

import csv
import math
import os
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import SetParameters

from perception_msgs.msg import StabilityStatus
from perception_tools.geometry import quat_to_matrix, wrap_angle

# Pad tops in slopes.world: name -> (x, y, z, roll).
PADS = {
    'flat': (0.0, 0.0, 0.0, 0.0),
    'slope_04': (10.0, 0.0, 0.449, 0.0698),
    'slope_08': (20.0, 0.0, 0.585, 0.1396),
    'slope_12': (30.0, 0.0, 0.719, 0.2094),
    'bumps': (-13.0, 0.0, 0.0, 0.0),
}

# name, pad, speed (m/s), yaw rate (rad/s), seconds at speed. A left turn on a pad
# turns towards the uphill side, which puts the outside of the turn downhill.
SCENARIOS = [
    ('flat_turn', 'flat', 1.0, 2.5, 2.6),
    ('slope_04_turn', 'slope_04', 1.0, 2.0, 3.2),
    ('slope_08_turn', 'slope_08', 1.2, 2.4, 3.0),
    ('slope_12_straight', 'slope_12', 0.6, 0.0, 3.0),
    ('slope_12_turn', 'slope_12', 0.8, 2.0, 3.2),
    ('bumps', 'bumps', 0.8, 0.0, 5.0),
]

RAMP_S = 1.5
SETTLE_S = 3.0
STOP_S = 1.5
RATE_HZ = 20.0
# Wheel contact points of the four-wheel variant, base frame.
WHEELS = np.array([[0.14, 0.17, 0.0], [0.14, -0.17, 0.0], [-0.14, 0.17, 0.0], [-0.14, -0.17, 0.0]])


def roll_pitch(R):
    return math.atan2(R[2, 1], R[2, 2]), math.asin(max(-1.0, min(1.0, -R[2, 0])))


class ScenarioRunner(Node):

    def __init__(self):
        super().__init__('stability_scenarios')
        self.output_dir = self.declare_parameter('output_dir', 'stability_results').value
        self.lift_height = self.declare_parameter('lift_height', 0.10).value
        self.repeats = self.declare_parameter('repeats', 3).value
        self.robot = self.declare_parameter('robot', 'ego_bot').value
        only = self.declare_parameter('scenarios', ['']).value
        self.scenarios = [sc for sc in SCENARIOS if sc[0] in only] if any(only) else SCENARIOS
        self.pose = None
        self.status = None
        self.speed = (0.0, 0.0)
        self.create_subscription(
            Odometry, '/odom',
            lambda m: setattr(self, 'speed', (m.twist.twist.linear.x, m.twist.twist.angular.z)), 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.on_states, 10)
        self.create_subscription(StabilityStatus, '/stability', self.on_status, 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_in', 10)
        self.set_state = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.set_params = self.create_client(SetParameters, '/collision_monitor/set_parameters')

    def on_states(self, msg):
        if self.robot in msg.name:
            self.pose = msg.pose[msg.name.index(self.robot)]

    def on_status(self, msg):
        self.status = msg

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def call(self, client, request):
        if not client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError(f'{client.srv_name} not available')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        return future.result()

    def set_gate(self, on):
        req = SetParameters.Request()
        req.parameters = [Parameter('stability_gate', value=on).to_parameter_msg()]
        self.call(self.set_params, req)

    def place(self, pad):
        x, y, z, roll = PADS[pad]
        s = EntityState()
        s.name = self.robot
        s.reference_frame = 'world'
        s.pose.position.x, s.pose.position.y, s.pose.position.z = x, y, z + 0.01
        s.pose.orientation.x = math.sin(roll / 2.0)
        s.pose.orientation.w = math.cos(roll / 2.0)
        self.call(self.set_state, SetEntityState.Request(state=s))

    def run(self, name, pad, speed, yaw_rate, turn_s, gate, rep):
        self.cmd_pub.publish(Twist())
        # Twice: placing a robot that is still tumbling leaves it moving.
        self.place(pad)
        self.spin_for(1.0)
        self.place(pad)
        self.spin_for(SETTLE_S)
        roll_pad = PADS[pad][3]
        normal = np.array([0.0, -math.sin(roll_pad), math.cos(roll_pad)])
        surface = np.array(PADS[pad][:3])
        samples = []
        t0 = time.time()
        total = RAMP_S + turn_s + STOP_S
        next_cmd = 0.0
        while (t := time.time() - t0) < total:
            rclpy.spin_once(self, timeout_sec=0.005)
            if t < next_cmd:
                continue
            next_cmd += 1.0 / RATE_HZ
            cmd = Twist()
            if t < RAMP_S:
                cmd.linear.x = speed * t / RAMP_S
            elif t < RAMP_S + turn_s:
                cmd.linear.x, cmd.angular.z = speed, yaw_rate
            self.cmd_pub.publish(cmd)
            if self.pose is None or self.status is None:
                continue
            q = self.pose.orientation
            R = np.array(quat_to_matrix((q.x, q.y, q.z, q.w)))
            p = self.pose.position
            pos = np.array([p.x, p.y, p.z])
            wheel_h = ((pos + WHEELS @ R.T) - surface) @ normal
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[:, 2] @ normal)))))
            true_roll, true_pitch = roll_pitch(R)
            st = self.status
            samples.append({
                'scenario': name, 'gate': int(gate), 'run': rep, 't': round(t, 3),
                'cmd_v': cmd.linear.x, 'cmd_w': cmd.angular.z,
                'v': self.speed[0], 'w': self.speed[1],
                'x': p.x, 'y': p.y, 'lift_m': float(wheel_h.max()),
                'tilt_deg': tilt, 'true_roll': true_roll, 'true_pitch': true_pitch,
                'roll': st.roll, 'pitch': st.pitch,
                'margin_deg': math.degrees(st.margin),
                'predicted_deg': math.degrees(st.predicted_margin),
                'level': st.level, 'scale': st.command_scale,
            })
        self.cmd_pub.publish(Twist())
        return samples

    def summarize(self, name, gate, rep, samples):
        lift = next((s['t'] for s in samples if s['lift_m'] > self.lift_height), None)
        crit = next((s['t'] for s in samples if s['level'] >= 3), None)
        warn = next((s['t'] for s in samples if s['level'] >= 2), None)
        upright = [s for s in samples if lift is None or s['t'] < lift]
        roll_err = [math.degrees(abs(wrap_angle(s['roll'] - s['true_roll']))) for s in upright]
        pitch_err = [math.degrees(abs(s['pitch'] - s['true_pitch'])) for s in upright]
        return {
            'scenario': name, 'gate': 'on' if gate else 'off', 'run': rep,
            'tipped': lift is not None,
            'max_tilt_deg': round(max(s['tilt_deg'] for s in samples), 1),
            'max_lift_cm': round(100 * max(s['lift_m'] for s in samples), 1),
            'max_v': round(max(abs(s['v']) for s in samples), 2),
            'max_w': round(max(abs(s['w']) for s in samples), 2),
            'first_warn_s': warn, 'first_critical_s': crit, 'lift_s': lift,
            'lead_s': round(lift - crit, 2) if lift is not None and crit is not None else None,
            'critical_fraction': round(sum(s['level'] >= 3 for s in samples) / len(samples), 3),
            'min_margin_deg': round(min(s['margin_deg'] for s in upright), 1) if upright else None,
            'min_scale': round(min(s['scale'] for s in samples), 2),
            'attitude_err_deg': round(float(np.sqrt(np.mean(np.square(roll_err + pitch_err)))), 2)
            if upright else None,
        }

    def run_all(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.spin_for(2.0)
        all_samples, summary = [], []
        for gate in (False, True):
            self.set_gate(gate)
            for rep in range(self.repeats):
                for name, pad, speed, yaw_rate, turn_s in self.scenarios:
                    samples = self.run(name, pad, speed, yaw_rate, turn_s, gate, rep)
                    if not samples:
                        self.get_logger().error(f'{name}: no data; is /stability published?')
                        continue
                    row = self.summarize(name, gate, rep, samples)
                    self.get_logger().info(str(row))
                    all_samples += samples
                    summary.append(row)
        self.set_gate(True)
        self.write_csv(os.path.join(self.output_dir, 'samples.csv'), all_samples)
        self.write_csv(os.path.join(self.output_dir, 'runs.csv'), summary)
        print_summary(summary)

    @staticmethod
    def write_csv(path, rows):
        if not rows:
            return
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def print_summary(rows):
    print('scenario | gate | tipped | max wheel lift cm | warning lead s | '
          'time critical | attitude error deg')
    keys = []
    for r in rows:
        if (r['scenario'], r['gate']) not in keys:
            keys.append((r['scenario'], r['gate']))
    for name, gate in keys:
        group = [r for r in rows if r['scenario'] == name and r['gate'] == gate]
        leads = [r['lead_s'] for r in group if r['lead_s'] is not None]
        errs = [r['attitude_err_deg'] for r in group if r['attitude_err_deg'] is not None]
        print(' | '.join([
            name, gate,
            '{}/{}'.format(sum(r['tipped'] for r in group), len(group)),
            ' '.join(str(r['max_lift_cm']) for r in group),
            '{:.1f}'.format(min(leads)) if leads else '-',
            '{:.0%}'.format(np.mean([r['critical_fraction'] for r in group])),
            '{:.1f}'.format(np.median(errs)) if errs else '-',
        ]))


def main(argv=None):
    rclpy.init(args=argv)
    node = ScenarioRunner()
    try:
        node.run_all()
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
