"""Conveyor-to-conveyor transfer: a robot moves cartons from one belt to another.

The infeed belt behind the arm brings cartons up to a through-beam and
halts there; the arm comes down onto the carton, takes it with a vacuum
cup, swings round and sets it on the outfeed belt in front of it, which
carries it away to the end of the line. Before it sets a carton down the
arm checks a presence zone at the place spot — on a slow outfeed it waits
for the previous carton to clear, and on a slow infeed for the next carton
to arrive, so both belt speeds show up in the cycle time.

The three speeds are the demo's knobs:

    --robot-speed F    the arm's joint speed limits times F (1.0 = nominal)
    --infeed-speed V   infeed belt speed in m/s
    --outfeed-speed V  outfeed belt speed in m/s
    --boxes N          how many cartons to transfer

Every combination bakes into one deterministic timeline; the script prints
the cycle time and the throughput, writes the recording as USD, and with
``--studio`` opens it in the 3D studio.

Runs from a checkout with no downloads (primitive-geometry arm):

    python examples/basics/conveyor_transfer_demo.py [out.usda] [--studio]
        [--robot-speed 1.0] [--infeed-speed 0.25] [--outfeed-speed 0.25] [--boxes 4]
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import botrail as bt

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# --- the cell (metres; z = 0 is the shop floor) ------------------------
BASE_Z = 0.72  # the pedestal's top: the arm's mounting plane
BELT_TOP = 0.72  # both belts convey at the arm's mounting height
BELT_WIDTH = 0.20
IN_Y, OUT_Y = 0.42, -0.42  # infeed behind the arm, outfeed in front of it
IN_X1 = 0.30  # the infeed ends just past the pick; it is as long as the queue
OUT_X0, OUT_X1 = -0.30, 1.60  # the outfeed starts just before the place
PICK_X = PLACE_X = 0.0  # both stations straight across from the arm
BOX = 0.08  # an 80 mm carton
BOX_PITCH = 0.30  # spacing of the cartons queued on the infeed
FIRST_BOX_X = -0.45
BEAM_RADIUS = 0.005
TOOL_FACE = 0.04  # the suction face, this far beyond `tool0`
HOVER = 0.12  # how far above a station the arm comes over before going down
PICK_GAP = 0.002  # the cup stops this far above the carton's top
SEAT_GAP = 0.004  # a carried carton is set down this far clear of the belt
WRIST = math.pi / 2  # the wrist roll in every taught pose (off its singularity)
READY = [0.0, -0.3, 2.0, math.pi - 1.7, WRIST, 0.0]
GRIP_TIME = 0.25  # vacuum build-up / release, s


def scaled_arm(robot_speed: float) -> bt.Robot:
    """The demo arm with every joint's velocity limit multiplied by
    `robot_speed`: the planner times each move from those limits, so this
    is the robot's speed override."""
    if robot_speed <= 0:
        raise ValueError("--robot-speed must be positive")
    urdf = (ASSETS / "simple_arm.urdf").read_text()
    urdf = re.sub(r'velocity="([0-9.]+)"', lambda m: f'velocity="{float(m[1]) * robot_speed:g}"', urdf)
    return bt.Robot.from_urdf_string(urdf)


def box_name(i: int) -> str:
    return f"carton_{i + 1}"


def build_cell(robot_speed: float = 1.0, infeed_speed: float = 0.25, outfeed_speed: float = 0.25,
               boxes: int = 4) -> bt.Scene:
    """The arm on its pedestal between the two belts, the cartons queued on
    the infeed, a through-beam at the pick and a presence zone at the place
    — and the motions taught by IK against them."""
    if infeed_speed <= 0 or outfeed_speed <= 0:
        raise ValueError("belt speeds must be positive")
    if boxes < 1:
        raise ValueError("--boxes must be at least 1")

    scene = bt.Scene(scaled_arm(robot_speed))
    bt.parts.pedestal(scene, "pedestal", height=BASE_Z, position=(0.0, 0.0), top=(0.24, 0.24), base=(0.40, 0.40),
                      column=0.14, yaw=math.pi / 2)
    scene.set_robot_base_pose(*scene.frame("pedestal/mount"))
    scene.allow_link_obstacle_contact("base_link", "pedestal/top")

    # Both belts run +x. The infeed is stopped at the start; its program
    # starts it and halts it each time a carton reaches the beam. It starts
    # a little upstream of the last carton in the queue.
    in_x0 = FIRST_BOX_X - (boxes - 1) * BOX_PITCH - BOX / 2 - 0.10
    bt.parts.conveyor(scene, "infeed", length=IN_X1 - in_x0, width=BELT_WIDTH,
                      position=((in_x0 + IN_X1) / 2, IN_Y, BELT_TOP), speed=infeed_speed, running=False)
    bt.parts.conveyor(scene, "outfeed", length=OUT_X1 - OUT_X0, width=BELT_WIDTH,
                      position=((OUT_X0 + OUT_X1) / 2, OUT_Y, BELT_TOP), speed=outfeed_speed, running=False)

    names = [box_name(i) for i in range(boxes)]
    for i, name in enumerate(names):
        bt.parts.carton(scene, name, size=(BOX, BOX, BOX),
                        position=(FIRST_BOX_X - i * BOX_PITCH, IN_Y, BELT_TOP + 1e-4))

    # At the end of the outfeed the cartons leave the cell: a sink there
    # returns them to the source's magazine under the floor. The source is
    # never started — the cartons begin on the infeed, already loaded.
    scene.add_source("magazine", pool=names, park=(OUT_X1 + 0.5, OUT_Y, -1.0), pitch=(0.0, 0.0, -BOX * 1.5),
                     position=(in_x0 + BOX, IN_Y, BELT_TOP + BOX / 2))
    scene.add_sink("line_end", zone_position=(OUT_X1 - 0.06, OUT_Y, BELT_TOP + BOX / 2),
                   zone_size=(0.12, BELT_WIDTH, BOX), source="magazine")

    # The pick beam trips on a carton's leading face, so it stands half a
    # carton downstream of the pick: a halted carton sits centred under the
    # cup. On the outfeed a presence zone covers the place spot, a carton's
    # length either side of it, and reads "busy" until the last carton set
    # down there has moved a carton clear.
    beam_z = BELT_TOP + BOX / 2
    bt.parts.photoelectric(scene, "at_pick", frm=(PICK_X + BOX / 2 + BEAM_RADIUS, IN_Y + 0.17, beam_z),
                           to=(PICK_X + BOX / 2 + BEAM_RADIUS, IN_Y - 0.17, beam_z), watch=names)
    scene.add_zone_sensor("place_busy", position=(PLACE_X, OUT_Y, BELT_TOP + BOX / 4),
                          size=(2 * BOX + 0.02, BELT_WIDTH, BOX / 2), watch=names)
    bt.parts.marking(scene, "floor/keep_out", rect=(-0.55, -0.70, 0.55, 0.70), dash=(0.20, 0.10), width=0.05)

    teach(scene)
    return scene


def _down(x: float, y: float) -> tuple[float, float, float, float]:
    """The tool pointing at the floor over (x, y), turned so the arm reaches
    it with its wrist roll at `WRIST`."""
    half = (math.atan2(y, x) + math.pi - WRIST) / 2
    return (math.cos(half), math.sin(half), 0.0, 0.0)


def teach(scene: bt.Scene) -> None:
    """Teach the stations by IK, each as a hover and a set-down, and the
    moves between them: the arm goes straight down and up at a station and
    swings across between them at hover height."""

    def at(x: float, y: float, face_z: float) -> list[float]:
        scene.set_joint_positions([math.atan2(y, x) - math.pi / 2, 0.5, 1.2, math.pi - 1.7, WRIST, 0.0])
        ik = scene.set_tcp_target((x, y, face_z + TOOL_FACE), _down(x, y))
        if not ik.converged:
            raise RuntimeError(f"the cell layout is out of reach at ({x:.3f}, {y:.3f}, {face_z:.3f})")
        return list(scene.joint_positions)

    pick_z = BELT_TOP + BOX + PICK_GAP
    place_z = BELT_TOP + BOX + SEAT_GAP
    over_pick = at(PICK_X, IN_Y, pick_z + HOVER)
    over_place = at(PLACE_X, OUT_Y, place_z + HOVER)

    scene.add_segment("to_pick", goal=over_pick)
    scene.add_segment("pick_down", goal=at(PICK_X, IN_Y, pick_z), kind="cartesian_line")
    scene.add_segment("lift", goal=over_pick, kind="cartesian_line")
    scene.add_segment("swing", goal=over_place)
    scene.add_segment("place_down", goal=at(PLACE_X, OUT_Y, place_z), kind="cartesian_line")
    scene.add_segment("clear", goal=over_place, kind="cartesian_line")
    scene.add_segment("home", goal=READY)
    scene.set_joint_positions(READY)


def author_sequences(scene: bt.Scene, boxes: int) -> list[str]:
    """Two programs on one scan cycle, as a PLC and a robot controller
    would run them.

    The infeed program runs the belt until a carton breaks the pick beam,
    halts it, and starts it again once the arm has lifted that carton out
    of the beam — so the next carton is on its way while the arm swings.

    The robot program, per carton: come over the pick and wait for a carton
    there, go down, grip, lift, swing over the outfeed, wait for the place
    spot to be free, set the carton down, let go and come back up. The
    outfeed runs throughout."""
    feed = scene.sequence("infeed")
    for n in range(1, boxes + 1):
        feed.step(f"run {n}", actions=[bt.seq.start("infeed")], transition=bt.seq.signal("at_pick"))
        feed.step(f"hold {n}", actions=[bt.seq.stop("infeed")], transition=bt.seq.signal("at_pick", False))

    arm = scene.sequence("robot")
    arm.step("start", actions=[bt.seq.start("outfeed")], transition=bt.seq.immediately())
    for i in range(boxes):
        box, n = box_name(i), i + 1
        arm.step(f"to pick {n}", actions=[bt.seq.motion("to_pick")])
        arm.step(f"await carton {n}", transition=bt.seq.signal("at_pick"))
        arm.step(f"pick {n}", actions=[bt.seq.motion("pick_down")])
        arm.step(f"grip {n}", actions=[bt.seq.attach(box)], transition=bt.seq.elapsed(GRIP_TIME))
        arm.step(f"lift {n}", actions=[bt.seq.motion("lift")])
        arm.step(f"swing {n}", actions=[bt.seq.motion("swing")])
        arm.step(f"await place {n}", transition=bt.seq.signal("place_busy", False))
        arm.step(f"place {n}", actions=[bt.seq.motion("place_down")])
        arm.step(f"release {n}", actions=[bt.seq.detach(box)], transition=bt.seq.elapsed(GRIP_TIME))
        arm.step(f"clear {n}", actions=[bt.seq.motion("clear")])
    arm.step("home", actions=[bt.seq.motion("home")])
    return [feed.name, arm.name]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", default="conveyor_transfer.usda", help="USD recording to write")
    ap.add_argument("--robot-speed", type=float, default=1.0, help="robot speed factor (1.0 = nominal)")
    ap.add_argument("--infeed-speed", type=float, default=0.25, help="infeed belt speed, m/s")
    ap.add_argument("--outfeed-speed", type=float, default=0.25, help="outfeed belt speed, m/s")
    ap.add_argument("--boxes", type=int, default=4, help="number of cartons to transfer")
    ap.add_argument("--studio", action="store_true", help="open the 3D studio on the bake")
    args = ap.parse_args()

    scene = build_cell(args.robot_speed, args.infeed_speed, args.outfeed_speed, args.boxes)
    names = author_sequences(scene, args.boxes)
    tl = scene.simulate_sequences(names, max_duration=600.0)

    print(f"robot speed x{args.robot_speed:g}, infeed {args.infeed_speed:g} m/s, "
          f"outfeed {args.outfeed_speed:g} m/s, {args.boxes} carton(s)")
    print(f"cycle time: {tl.duration:.2f} s")
    if args.boxes > 1:
        first = tl.step_span("robot/pick 1").start
        per_box = (tl.step_span(f"robot/pick {args.boxes}").start - first) / (args.boxes - 1)
        print(f"pick to pick: {per_box:.2f} s per carton -> {3600 / per_box:.0f} cartons/hour")
    # Where the arm stood idle: waiting for the next carton to reach the
    # pick (a slow infeed) or for the place spot to clear (a slow outfeed).
    span = lambda step: tl.step_span(f"robot/{step}").duration  # noqa: E731
    print(f"  arm waited on the infeed:  {sum(span(f'await carton {n}') for n in range(2, args.boxes + 1)):.2f} s")
    print(f"  arm waited on the outfeed: {sum(span(f'await place {n}') for n in range(1, args.boxes + 1)):.2f} s")

    out = Path(args.out)
    for w in tl.export_usd(out, fps=60.0):
        print(f"warning: {w}")
    print(f"exported to {out}")

    if args.studio:
        bt.studio(scene)


if __name__ == "__main__":
    main()
