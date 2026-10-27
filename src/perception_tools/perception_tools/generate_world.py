#!/usr/bin/env python3
"""Generate a randomized warehouse world for Gazebo.

Each seed gives a different layout of shelves and pallets plus pedestrians and
forklifts moving between free points. Writes the world and a matching
.agents.yaml with the paths for agent_mover. Used to build separate train / val /
test scenarios so evaluation runs on layouts the detector has not seen.

    ros2 run perception_tools generate_world --seed 7 --output worlds/scene_007.world
"""

import argparse
import math
import os
import random

WALL_T = 0.2
WALL_H = 2.5
SHELF_SIZES = [(4.0, 1.0), (5.0, 1.0), (6.0, 1.0)]
PALLET_SIZE = (1.2, 0.8)
SHELF_GAP = 1.4      # minimum aisle width between shelves (m)
ACTOR_MARGIN = 0.6   # clearance kept between actor paths and obstacles (m)


def _overlaps(a, b, gap):
    ax, ay, asx, asy = a
    bx, by, bsx, bsy = b
    return (abs(ax - bx) < (asx + bsx) / 2.0 + gap and
            abs(ay - by) < (asy + bsy) / 2.0 + gap)


def _inside(px, py, box, margin):
    cx, cy, sx, sy = box
    return abs(px - cx) < sx / 2.0 + margin and abs(py - cy) < sy / 2.0 + margin


class Layout:

    def __init__(self, seed, size_x, size_y, spawn, clear_radius):
        self.rng = random.Random(seed)
        self.size_x = size_x
        self.size_y = size_y
        self.spawn = spawn
        self.clear_radius = clear_radius
        self.shelves = []
        self.pallets = []
        self.actors = []   # (name, kind, (ax, ay), (bx, by), speed)

    @property
    def obstacles(self):
        return self.shelves + self.pallets

    def _near_spawn(self, box):
        sx, sy = self.spawn
        cx, cy, bx, by = box
        dx = max(abs(sx - cx) - bx / 2.0, 0.0)
        dy = max(abs(sy - cy) - by / 2.0, 0.0)
        return math.hypot(dx, dy) < self.clear_radius

    def _place(self, size, existing, gap, tries=200):
        hx, hy = self.size_x / 2.0, self.size_y / 2.0
        for _ in range(tries):
            sx, sy = size if self.rng.random() < 0.5 else (size[1], size[0])
            cx = self.rng.uniform(-hx + sx / 2.0 + 1.0, hx - sx / 2.0 - 1.0)
            cy = self.rng.uniform(-hy + sy / 2.0 + 1.0, hy - sy / 2.0 - 1.0)
            box = (cx, cy, sx, sy)
            if self._near_spawn(box):
                continue
            if any(_overlaps(box, o, gap) for o in existing):
                continue
            return box
        return None

    def add_obstacles(self, n_shelves, n_pallets):
        for _ in range(n_shelves):
            box = self._place(self.rng.choice(SHELF_SIZES), self.obstacles, SHELF_GAP)
            if box:
                self.shelves.append(box)
        for _ in range(n_pallets):
            box = self._place(PALLET_SIZE, self.obstacles, 0.8)
            if box:
                self.pallets.append(box)

    def _free_point(self):
        hx, hy = self.size_x / 2.0 - 1.0, self.size_y / 2.0 - 1.0
        for _ in range(200):
            p = (self.rng.uniform(-hx, hx), self.rng.uniform(-hy, hy))
            if math.hypot(p[0] - self.spawn[0], p[1] - self.spawn[1]) < self.clear_radius:
                continue
            if any(_inside(p[0], p[1], o, ACTOR_MARGIN) for o in self.obstacles):
                continue
            return p
        return None

    def _clear_path(self, a, b):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(2, int(length / 0.2))
        for i in range(steps + 1):
            t = i / steps
            x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            if any(_inside(x, y, o, ACTOR_MARGIN) for o in self.obstacles):
                return False
        return True

    def add_actors(self, kind, count, speed_range, min_length=4.0):
        prefix = 'pedestrian' if kind == 'pedestrian' else 'forklift'
        for i in range(count):
            for _ in range(200):
                a, b = self._free_point(), self._free_point()
                if a is None or b is None:
                    continue
                if math.hypot(b[0] - a[0], b[1] - a[1]) < min_length:
                    continue
                if not self._clear_path(a, b):
                    continue
                speed = self.rng.uniform(*speed_range)
                self.actors.append((f'{prefix}_{i + 1}', kind, a, b, speed))
                break


def _box_model(name, box, height, color):
    cx, cy, sx, sy = box
    geom = f'<box><size>{sx:.2f} {sy:.2f} {height:.2f}</size></box>'
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{cx:.2f} {cy:.2f} {height / 2.0:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry>{geom}</geometry></collision>
        <visual name="v"><geometry>{geom}</geometry>
          <material><ambient>{color} 1</ambient><diffuse>{color} 1</diffuse></material>
        </visual>
      </link>
    </model>
"""


def _walls(size_x, size_y):
    hx, hy = size_x / 2.0, size_y / 2.0
    parts = [
        ('px', hx, 0.0, WALL_T, size_y + WALL_T),
        ('nx', -hx, 0.0, WALL_T, size_y + WALL_T),
        ('py', 0.0, hy, size_x + WALL_T, WALL_T),
        ('ny', 0.0, -hy, size_x + WALL_T, WALL_T),
    ]
    out = ['    <model name="walls">\n      <static>true</static>\n      <link name="link">\n']
    for tag, x, y, sx, sy in parts:
        geom = f'<box><size>{sx:.2f} {sy:.2f} {WALL_H}</size></box>'
        pose = f'<pose>{x:.2f} {y:.2f} {WALL_H / 2.0} 0 0 0</pose>'
        out.append(f'        <collision name="wall_{tag}_c">{pose}<geometry>{geom}</geometry>'
                   f'</collision>\n')
        out.append(f'        <visual name="wall_{tag}_v">{pose}<geometry>{geom}</geometry>'
                   f'<material><ambient>0.85 0.85 0.85 1</ambient></material></visual>\n')
    out.append('      </link>\n    </model>\n')
    return ''.join(out)


# Moving objects are seen by the LiDAR but exert no contact forces; as kinematic
# bodies they would otherwise shove the robot around.
NO_CONTACT = ('<surface><contact><collide_without_contact>true'
              '</collide_without_contact></contact></surface>')

PERSON_LINK = (
    '        <collision name="torso_c"><pose>0 0 0.75 0 0 0</pose><geometry><cylinder>'
    '<radius>0.22</radius><length>1.5</length></cylinder></geometry>' + NO_CONTACT +
    '</collision>\n'
    '        <visual name="torso_v"><pose>0 0 0.75 0 0 0</pose><geometry><cylinder>'
    '<radius>0.22</radius><length>1.5</length></cylinder></geometry>'
    '<material><ambient>0.2 0.6 0.2 1</ambient></material></visual>\n'
    '        <collision name="head_c"><pose>0 0 1.65 0 0 0</pose><geometry><sphere>'
    '<radius>0.13</radius></sphere></geometry>' + NO_CONTACT + '</collision>\n'
    '        <visual name="head_v"><pose>0 0 1.65 0 0 0</pose><geometry><sphere>'
    '<radius>0.13</radius></sphere></geometry>'
    '<material><ambient>0.9 0.75 0.6 1</ambient></material></visual>\n')

FORKLIFT_LINK = (
    '        <collision name="body_c"><pose>0 0 0.7 0 0 0</pose><geometry><box>'
    '<size>1.2 0.8 1.4</size></box></geometry>' + NO_CONTACT + '</collision>\n'
    '        <visual name="body_v"><pose>0 0 0.7 0 0 0</pose><geometry><box>'
    '<size>1.2 0.8 1.4</size></box></geometry>'
    '<material><ambient>0.8 0.5 0.1 1</ambient></material></visual>\n')


def _agent_model(name, kind, a, b):
    yaw = math.atan2(b[1] - a[1], b[0] - a[0])
    body = PERSON_LINK if kind == 'pedestrian' else FORKLIFT_LINK
    return (f'    <model name="{name}">\n'
            f'      <pose>{a[0]:.2f} {a[1]:.2f} 0 0 0 {yaw:.4f}</pose>\n'
            f'      <link name="body">\n'
            f'        <gravity>false</gravity>\n        <kinematic>true</kinematic>\n'
            f'{body}      </link>\n    </model>\n')


def agents_yaml(layout):
    """Paths for agent_mover, matching the models in the generated world."""
    lines = ['agents:']
    for name, _, a, b, speed in layout.actors:
        lines.append(f'  - name: {name}')
        lines.append(f'    speed: {speed:.2f}')
        lines.append('    mode: pingpong')
        lines.append(f'    waypoints: [[{a[0]:.2f}, {a[1]:.2f}], [{b[0]:.2f}, {b[1]:.2f}]]')
    return '\n'.join(lines) + '\n'


def build_world(layout):
    parts = [
        '<?xml version="1.0"?>\n<sdf version="1.6">\n  <world name="warehouse">\n',
        '    <include><uri>model://sun</uri></include>\n',
        '    <include><uri>model://ground_plane</uri></include>\n',
        '    <scene><ambient>0.6 0.6 0.6 1</ambient>'
        '<background>0.7 0.75 0.8 1</background></scene>\n',
        '    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">\n'
        '      <ros><namespace>/gazebo</namespace></ros>\n'
        '      <update_rate>50.0</update_rate>\n    </plugin>\n',
        _walls(layout.size_x, layout.size_y),
    ]
    for i, box in enumerate(layout.shelves):
        parts.append(_box_model(f'shelf_{i + 1}', box, 2.0, '0.6 0.5 0.35'))
    for i, box in enumerate(layout.pallets):
        parts.append(_box_model(f'pallet_{i + 1}', box, 0.6, '0.55 0.45 0.3'))
    for name, kind, a, b, _ in layout.actors:
        parts.append(_agent_model(name, kind, a, b))
    parts.append('  </world>\n</sdf>\n')
    return ''.join(parts)


def generate(seed, size_x=20.0, size_y=14.0, shelves=4, pallets=3,
             pedestrians=2, vehicles=1, spawn=(-8.0, 0.0), clear_radius=1.5):
    layout = Layout(seed, size_x, size_y, spawn, clear_radius)
    layout.add_obstacles(shelves, pallets)
    layout.add_actors('pedestrian', pedestrians, (0.9, 1.4))
    layout.add_actors('vehicle', vehicles, (1.0, 2.0), min_length=6.0)
    return build_world(layout), layout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--size-x', type=float, default=20.0)
    parser.add_argument('--size-y', type=float, default=14.0)
    parser.add_argument('--shelves', type=int, default=4)
    parser.add_argument('--pallets', type=int, default=3)
    parser.add_argument('--pedestrians', type=int, default=2)
    parser.add_argument('--vehicles', type=int, default=1)
    parser.add_argument('--spawn-x', type=float, default=-8.0)
    parser.add_argument('--spawn-y', type=float, default=0.0)
    args = parser.parse_args(argv)

    world, layout = generate(
        args.seed, args.size_x, args.size_y, args.shelves, args.pallets,
        args.pedestrians, args.vehicles, (args.spawn_x, args.spawn_y))
    with open(args.output, 'w') as fh:
        fh.write(world)
    agents_path = os.path.splitext(args.output)[0] + '.agents.yaml'
    with open(agents_path, 'w') as fh:
        fh.write(agents_yaml(layout))
    print(f'wrote {args.output} and {agents_path}: {len(layout.shelves)} shelves, '
          f'{len(layout.pallets)} pallets, {len(layout.actors)} moving objects')


if __name__ == '__main__':
    main()
