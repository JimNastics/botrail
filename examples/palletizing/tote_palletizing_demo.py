"""A tote palletizer: a robot takes totes off one conveyor and stacks them on
a pallet riding a second conveyor at right angles to it.

The cell is an L round one 4-axis palletizer on a welded riser:

  * the tote infeed — an Interroll MCP roller line running towards the
    robot, 600 x 400 x 220 Euro totes on it one pitch apart, a photo-eye at
    the pick end. It indexes one tote at a time, and a tote for every one
    the robot takes comes in at the far end;
  * the pallet line — Interroll MPP pallet rollers at right angles to the
    infeed: an infeed section with the next empty pallet waiting, the
    palletizing station with a beam through the deck boards, and an
    outfeed. An EPAL 1 (1200 x 800) takes four totes side by side, 2 x 2,
    and the stack goes up to six layers — 24 totes, about 2 m high;
  * the robot — a FANUC M-410iC/185 from the catalog (`--robot builtin`
    draws a generic 4-axis palletizer from primitives instead, no download)
    with an integrator-built tote gripper: a frame that fits inside the
    tote's rim and four fingers that spread against its inner walls.

What runs: the empty pallet rolls into the station; the robot waits for it,
then takes tote after tote from the infeed's stop and sets it down, layer by
layer, the far slots of each layer first. With the sixth layer complete it
parks and raises `pallet_full`; the line sends the full pallet out and draws
the next empty one into the station. Every hand-off is a signal, the tote by
its photo-eye and the pallet by its beam, and every planned move of the arm
— tote in the gripper included — is checked against the cell.

Run with:  python examples/palletizing/tote_palletizing_demo.py [out.usdc] [--studio]
                        [--robot catalog|builtin] [--layers N] [--catalog-root DIR]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import botrail as bt  # noqa: E402
import _palletizing_equipment as eq  # noqa: E402

# ---- what is ordered -------------------------------------------------
PALLETIZER = "fanuc/m410ic/m410ic-185/r2"
CATALOG_ROOT: Path | None = None

# ---- the goods -------------------------------------------------------
TOTE = (0.60, 0.40, 0.22)             # Euro container 600 x 400 x 220
TOTE_KG = 12.0
TOTE_BLUE = (0.05, 0.22, 0.55)
PALLET = (1.2, 0.8, 0.144)            # EPAL 1
PER_LAYER = 4                         # 2 x 2 on a 1200 x 800 pallet
LAYERS = 6                            # stacked up to six high
SEAT = 0.004                          # what is set down stands this proud of what it stands on
GAP = 0.015                           # between totes in a layer

# ---- the layout (metres) -----------------------------------------------
ROBOT_XY, RISER = (0.0, 0.0), 0.80
ROBOT_YAW = 3 * math.pi / 4           # facing between the infeed (-x) and the pallet (+y)
TOTE_TOP = 0.75                       # tote infeed: roller tops
INFEED_X = -1.75                      # the infeed runs +y along this line ...
INFEED_END = 0.30                     # ... and ends here
INFEED_LEN = 4.4
PITCH = 0.80                          # tote spacing on the infeed
ON_BELT = 4                           # totes on the infeed at the start
PAL_TOP = 0.50                        # pallet line: roller tops
PAL_Y = 1.75                          # the pallet line runs +x along this line
SECTIONS = (("pallet_in", -3.90, -0.75), ("station", -0.75, 0.75), ("pallet_out", 0.75, 3.90))
PALLET_X = (-1.50, -3.25)             # the first pallet and the next, on the infeed section

# ---- motion ------------------------------------------------------------
TOTE_SPEED = 0.4                      # m/s, the tote infeed
PALLET_SPEED = 0.3                    # m/s, the pallet line
HOVER = 0.30                          # approach and retract above a pick or a place
READY = [0.0, 0.0, 0.0, 0.0]
JOINTS = ("J1", "J2", "J3", "J4")
SPEEDS = [math.radians(v) for v in (140, 140, 140, 305)]   # M-410iC/185 axis speeds

STUDIO_VIEW = ((4.6, -4.2, 4.4), (-0.4, 0.9, 0.8))

PICK = (INFEED_X, INFEED_END - 0.02 - TOTE[0] / 2)    # a tote stopped on the eye


# ================================================================ the machines
def _link(name: str, body: str) -> str:
    return f'<link name="{name}">{body}</link>'


def _joint(name: str, parent: str, child: str, xyz, *, axis="0 1 0", rpy="0 0 0", lo=-3.14, hi=3.14,
           speed=2.4, mimic: str | None = None) -> str:
    m = f'<mimic joint="{mimic}" multiplier="-1"/>' if mimic else ""
    return (f'<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy}"/><axis xyz="{axis}"/>'
            f'<limit lower="{lo}" upper="{hi}" effort="5000" velocity="{speed:.3f}"/>{m}</joint>')


def builtin_palletizer_urdf() -> str:
    """A generic 4-axis palletizer drawn from primitives, in the class of the
    M-410: J1 turns the column, J2 leans the upper arm, J3 sets the forearm's
    angle to the floor (as a parallel link does — the elbow carries a mimic
    joint that cancels J2), a second mimic at the wrist keeps the flange
    level, and J4 turns the flange. Root `base_link` on the floor, the tool
    frame `flange` with +Z pointing down."""
    paint, dark = (0.90, 0.42, 0.05), (0.12, 0.12, 0.13)
    box, cyl = eq._box_xml, eq._cyl_xml
    l2, l3, drop = 1.25, 1.55, 0.30
    hpi = math.pi / 2
    links = [
        _link("base_link", cyl(0.36, 0.12, (0, 0, 0.06), color=dark) + box((0.70, 0.70, 0.10), (0, 0, 0.05), color=dark)),
        _link("turret", box((0.70, 0.56, 0.48), (0.08, 0, 0.30), color=paint)
              + cyl(0.18, 0.62, (0.30, 0, 0.62), (hpi, 0, 0), dark)),
        _link("upper_arm", box((0.24, 0.28, l2 - 0.30), (0, 0, l2 / 2), color=paint)),
        _link("elbow", cyl(0.16, 0.36, (0, 0, 0), (hpi, 0, 0), dark)),
        _link("forearm", box((l3 - 0.30, 0.22, 0.22), (l3 / 2, 0, 0), color=paint)
              + box((0.30, 0.30, 0.30), (-0.25, 0, 0), color=dark, collide=False)),
        _link("wrist", box((0.20, 0.20, drop - 0.06), (0, 0, -drop / 2 + 0.03), color=paint)),
        _link("flange", cyl(0.12, 0.03, (0, 0, -0.015), color=dark, collide=False)),
    ]
    joints = [
        _joint("J1", "base_link", "turret", (0, 0, 0.12), axis="0 0 1", lo=-3.14, hi=3.14),
        _joint("J2", "turret", "upper_arm", (0.30, 0, 0.62), lo=-0.90, hi=1.40),
        _joint("J2m", "upper_arm", "elbow", (0, 0, l2), lo=-3.14, hi=3.14, mimic="J2"),
        _joint("J3", "elbow", "forearm", (0, 0, 0), lo=-1.30, hi=1.40),
        _joint("J3m", "forearm", "wrist", (l3, 0, 0), lo=-3.14, hi=3.14, mimic="J3"),
        _joint("J4", "wrist", "flange", (0, 0, -drop), axis="0 0 1", rpy=f"{math.pi} 0 0", lo=-6.28, hi=6.28,
               speed=5.3),
    ]
    return f'<?xml version="1.0"?><robot name="palletizer">{"".join(links)}{"".join(joints)}</robot>'


GRIP_LENGTH = 0.22   # flange face to the gripper frame's underside: the tote's rim


def tote_gripper_urdf(name: str = "eoat") -> str:
    """The tote gripper an integrator builds: an adapter plate, a column, an
    aluminium frame just inside the tote's rim, and four fingers that reach
    down inside the tote and spread against its walls. Root at the flange
    face, +Z into the tool; `<name>_tcp` at the frame's underside, which
    rests on the rim."""
    steel, alu, finger = (0.55, 0.57, 0.60), eq.ALUMINIUM, (0.20, 0.21, 0.23)
    body = eq._box_xml((0.24, 0.24, 0.02), (0, 0, 0.01), color=steel, collide=False)
    body += eq._box_xml((0.14, 0.14, 0.14), (0, 0, 0.09), color=alu, collide=False)
    body += eq._box_xml((0.56, 0.36, 0.04), (0, 0, 0.20), color=alu, collide=False)
    body += eq._box_xml((0.10, 0.06, 0.06), (0, 0.20, 0.17), color=(0.10, 0.10, 0.11), collide=False)
    for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        body += eq._box_xml((0.04, 0.10, 0.10), (sx * 0.25, sy * 0.10, 0.27), color=finger, collide=False)
    body += '<collision><origin xyz="0 0 0.12"/><geometry><box size="0.56 0.36 0.20"/></geometry></collision>'
    return (f'<?xml version="1.0"?><robot name="{name}">'
            f'<link name="{name}_mount">{body}</link><link name="{name}_tcp"/>'
            f'<joint name="{name}_tcp_joint" type="fixed"><parent link="{name}_mount"/><child link="{name}_tcp"/>'
            f'<origin xyz="0 0 {GRIP_LENGTH:.4f}"/></joint></robot>')


def palletizer(kind: str) -> bt.Robot:
    gripper = bt.Robot.from_urdf_string(tote_gripper_urdf("eoat"))
    if kind == "builtin":
        # the parallel-link elbow: the two arms overlap there by design
        arm = bt.Robot.from_urdf_string(builtin_palletizer_urdf()).allow_collisions([("upper_arm", "forearm")])
    elif CATALOG_ROOT:
        arm = bt.Robot.from_package(CATALOG_ROOT / PALLETIZER, catalog_root=CATALOG_ROOT)
    else:
        arm = bt.Robot.from_catalog(PALLETIZER)
    return arm.attach_tool(gripper, flange="flange" if kind == "builtin" else None, tcp="eoat_tcp")


# ================================================================ the goods
def names_under(scene: bt.Scene, prefix: str) -> list:
    return [n for n in scene.obstacle_names if n.startswith(prefix + "/")]


def tote(scene: bt.Scene, name: str, centre) -> str:
    """A Euro tote as one obstacle — so it attaches, rides and stacks as one
    thing — drawn as the library's tote: ribs, stacking rim, grips."""
    made = scene.add_box(name, size=TOTE, position=centre, quaternion=eq._quat(math.pi / 2), color=TOTE_BLUE)
    bt.parts.appearance(scene, made, "tote", TOTE, tint=True)
    scene.set_part(made, kind="obstacle", category="workpiece", model="Euro container 600x400x220",
                   mass_kg=TOTE_KG)
    return made


def deck_z() -> float:
    """The deck of a pallet standing on the pallet line."""
    return PAL_TOP + 0.002 + PALLET[2]


def slot(n: int) -> tuple[float, float, float]:
    """Where tote `n` of the pallet sets down: the TCP (the tote's rim
    centre). Four to a layer, 600 along the pallet's length and 400 across,
    column-stacked; in each layer the slots far from the robot first, so the
    gripper never reaches over a tote it has just set down."""
    layer, k = divmod(n, PER_LAYER)
    i, j = ((1, 1), (-1, 1), (1, -1), (-1, -1))[k]            # +y is away from the robot
    x = i * (TOTE[0] + GAP) / 2
    y = PAL_Y + j * (TOTE[1] + GAP) / 2
    return x, y, deck_z() + SEAT + layer * (TOTE[2] + SEAT) + TOTE[2]


# ================================================================ the cell
def floor(scene: bt.Scene) -> None:
    slab = scene.add_box("floor/slab", (40.0, 40.0, 0.05), (0.0, 0.0, -0.025), color=(0.78, 0.78, 0.77))
    scene.set_obstacle_enabled(slab, False)
    scene.set_obstacle_material(slab, metalness=0.0, roughness=0.9)
    # the yellow walkway line round the cell
    for k, (cx, cy, sx, sy) in enumerate(((-0.2, -4.6, 8.4, 0.08), (-0.2, 3.0, 8.4, 0.08),
                                          (-4.4, -0.8, 0.08, 7.6), (4.0, -0.8, 0.08, 7.6))):
        made = scene.add_box(f"floor/line{k}", (sx, sy, 0.002), (cx, cy, 0.001), color=eq.RAL_1023)
        scene.set_obstacle_enabled(made, False)


def tote_infeed(scene: bt.Scene, count: int) -> list:
    """The tote infeed with its photo-eye, the totes on it and the ones still
    to come (a source that sets one down at the far end per index). Returns
    every tote, in the order the robot will take them."""
    mid = INFEED_END - INFEED_LEN / 2
    eq.case_roller(scene, "infeed", INFEED_LEN, (INFEED_X, mid), direction=(0.0, 1.0), top=TOTE_TOP,
                   speed=TOTE_SPEED)
    stop = scene.add_box("infeed/end_stop", (eq.MCP_BF + 0.08, 0.04, 0.10), (INFEED_X, INFEED_END + 0.02,
                         TOTE_TOP + 0.05), color=eq.RAL_1023)
    scene.set_obstacle_enabled(stop, False)
    z = TOTE_TOP + SEAT + TOTE[2] / 2
    on_belt = [tote(scene, f"totes/t{k}", (INFEED_X, PICK[1] - k * PITCH, z)) for k in range(min(ON_BELT, count))]
    park = (INFEED_X, PICK[1] - ON_BELT * PITCH, -1.0)
    pool = [tote(scene, f"totes/t{k}", (park[0], park[1], park[2] - 0.3 * (k - ON_BELT)))
            for k in range(ON_BELT, count)]
    if pool:
        # The upstream line, off the picture: one tote at the far end per index.
        scene.add_source("tote_feed", pool=pool, park=park, pitch=(0.0, 0.0, -0.3),
                         position=(INFEED_X, park[1], z), interval=0.0, running=False)
    zb = z
    totes = on_belt + pool
    scene.add_beam_sensor("eye_tote", frm=(INFEED_X - 0.40, INFEED_END - 0.02, zb),
                          to=(INFEED_X + 0.40, INFEED_END - 0.02, zb), watch=totes)
    scene.set_part("eye_tote", manufacturer="SICK", model="WL12G-3B2531", category="sensor.photoelectric")
    return totes


def pallet_line(scene: bt.Scene) -> dict:
    """The pallet line at right angles to the infeed: an empty pallet in
    front of the station, the next one waiting, beams through the deck
    boards at the station and at the end of the outfeed."""
    for name, x0, x1 in SECTIONS:
        eq.pallet_roller(scene, f"line/{name}", x1 - x0, ((x0 + x1) / 2, PAL_Y), direction=(1.0, 0.0),
                         top=PAL_TOP, speed=PALLET_SPEED, zone_height=1.9)
    for side, s in (("f", -1), ("b", 1)):         # the station's pallet guides
        g = scene.add_box(f"line/station/guide_{side}", (1.40, 0.05, 0.12),
                          (0.0, PAL_Y + s * (eq.MPP_CW / 2 + 0.03), PAL_TOP + 0.10), color=eq.RAL_1023)
        scene.set_obstacle_enabled(g, False)
    pallets = {}
    for name, x in zip(("pallet1", "pallet2"), PALLET_X):
        bt.parts.pallet(scene, name, (x, PAL_Y, PAL_TOP + 0.002), size=PALLET, model="EPAL 1")
        pallets[name] = names_under(scene, name)
    boards = [n for names in pallets.values() for n in names]
    zb = PAL_TOP + 0.002 + PALLET[2] - 0.011
    station_eye = PALLET[0] / 2 + 0.01
    for name, x in (("eye_station", station_eye), ("eye_out", SECTIONS[2][2] - 0.05)):
        scene.add_beam_sensor(name, frm=(x, PAL_Y - 0.45, zb), to=(x, PAL_Y + 0.45, zb), watch=boards)
        scene.set_part(name, manufacturer="SICK", model="WL12G-3B2531", category="sensor.photoelectric")
    return pallets


def teach(scene: bt.Scene, count: int, seed: list) -> dict:
    """Every pose the palletizer works from, solved against its own
    kinematics: the gripper square to the tote, its long side along the
    infeed at the pick and along the pallet at the place, `HOVER` above for
    the approach. Each becomes a planned motion; the joint values are kept
    for the straight moves in and out."""
    lo, hi = scene.robot_of("robot").joint_limits[3]

    def down(yaw: float) -> tuple:
        # tool +Z at the floor, tool +X along `yaw` (x, y, z, w)
        return (math.cos(yaw / 2), math.sin(yaw / 2), 0.0, 0.0)

    taught = {}

    def pose(name: str, xyz, yaw: float, start: list) -> list:
        scene.set_joint_positions(start, robot="robot")
        ik = scene.set_tcp_target(xyz, down(yaw), robot="robot")
        if not ik.converged:
            raise RuntimeError(f"the palletizer cannot reach {name} at {tuple(round(v, 3) for v in xyz)}: "
                               f"{ik.pos_error * 1e3:.0f} mm short")
        q = list(scene.joint_positions_of("robot"))
        # The gripper is square to the tote either way round: take the
        # flange angle a half turn from IK's when that is the shorter swing.
        q[3] = min((q[3] + n * math.pi for n in range(-4, 5) if lo <= q[3] + n * math.pi <= hi),
                   key=lambda v: abs(v - start[3]))
        scene.add_segment(name, goal=q, robot="robot")
        taught[name] = q
        return q

    grip = TOTE_TOP + SEAT + TOTE[2]
    hi_q = pose("pick_hi", (*PICK, grip + HOVER), math.pi / 2, seed)
    pose("pick_lo", (*PICK, grip), math.pi / 2, hi_q)
    for n in range(count):
        x, y, top = slot(n)
        hq = pose(f"set{n}_hi", (x, y, top + HOVER), 0.0, hi_q)
        pose(f"set{n}_lo", (x, y, top), 0.0, hq)
    scene.set_joint_positions(seed, robot="robot")
    scene.add_segment("ready", goal=seed, robot="robot")
    return taught


def straight(frm: list, to: list, share: float = 0.6) -> dict:
    """The approach or retreat between a hover pose and the pick or place
    below it: a joint ramp at `share` of the axis speeds, 0.2 s to settle."""
    t = max(abs(b - a) / (share * v) for a, b, v in zip(frm, to, SPEEDS)) + 0.2
    return bt.seq.ramp(dict(zip(JOINTS, to)), round(t, 3), robot="robot")


# ================================================================ the programs
def programs(scene: bt.Scene, totes: list, q: dict) -> list:
    for sig in ("grip", "pallet_ready", "pallet_full"):
        scene.define_signal(sig, initial=False)
    count = len(totes)

    # ---- the tote infeed: index one tote onto the eye, hold it there until
    # the robot has taken it, and let the upstream line send the next.
    inf = scene.sequence("infeed")
    for k in range(count):
        inf.step(f"run{k}", actions=[bt.seq.start("infeed")], transition=bt.seq.signal("eye_tote"))
        held = [bt.seq.stop("infeed")]
        if k + ON_BELT < count:
            held.append(bt.seq.start("tote_feed"))
        inf.step(f"hold{k}", actions=held, transition=bt.seq.signal("eye_tote", False))

    # ---- the pallet line: the empty pallet into the station, then — once
    # it is full — out, and the next empty one in behind it.
    pl = scene.sequence("pallets")
    pl.step("feed", actions=[bt.seq.start("line/pallet_in"), bt.seq.start("line/station")],
            transition=bt.seq.signal("eye_station"))
    pl.step("in_station", actions=[bt.seq.stop("line/pallet_in"), bt.seq.stop("line/station"),
                                   bt.seq.set_signal("pallet_ready")],
            transition=bt.seq.signal("pallet_full"))
    pl.step("send_out", actions=[bt.seq.set_signal("pallet_ready", False)]
            + [bt.seq.start(f"line/{n}") for n, _, _ in SECTIONS],
            transition=bt.seq.signal("eye_station", False))           # the full pallet's tail
    pl.step("next_in", transition=bt.seq.signal("eye_station"))        # the next one's head
    pl.step("next_ready", actions=[bt.seq.stop("line/pallet_in"), bt.seq.stop("line/station")],
            transition=bt.seq.signal("eye_out"))
    pl.step("out", actions=[bt.seq.stop("line/pallet_out")])

    # ---- the robot: tote by tote, layer by layer --------------------------
    rb = scene.sequence("robot")
    rb.step("await_pallet", transition=bt.seq.signal("pallet_ready"))
    for k, name in enumerate(totes):
        rb.step(f"wait{k}", transition=bt.seq.signal("eye_tote"))
        rb.step(f"over{k}", actions=[bt.seq.motion("pick_hi")])
        rb.step(f"down{k}", actions=[straight(q["pick_hi"], q["pick_lo"])])
        rb.step(f"grip{k}", actions=[bt.seq.attach(name, touch_links=["eoat_mount"], robot="robot"),
                                     bt.seq.set_signal("grip")], transition=bt.seq.elapsed(0.3))
        rb.step(f"lift{k}", actions=[straight(q["pick_lo"], q["pick_hi"])])
        rb.step(f"swing{k}", actions=[bt.seq.motion(f"set{k}_hi")])
        rb.step(f"place{k}", actions=[straight(q[f"set{k}_hi"], q[f"set{k}_lo"])])
        rb.step(f"release{k}", actions=[bt.seq.detach(name), bt.seq.set_signal("grip", False)],
                transition=bt.seq.elapsed(0.3))
        rb.step(f"clear{k}", actions=[straight(q[f"set{k}_lo"], q[f"set{k}_hi"])])
    rb.step("park", actions=[bt.seq.motion("ready")])
    rb.step("full", actions=[bt.seq.set_signal("pallet_full")])
    return ["infeed", "pallets", "robot"]


# ================================================================ assembly
def build(robot: str = "catalog", layers: int = LAYERS) -> tuple[bt.Scene, dict]:
    if not 1 <= layers <= LAYERS:
        raise ValueError(f"the pallet takes 1 to {LAYERS} layers, not {layers}")
    count = layers * PER_LAYER
    scene = bt.Scene(palletizer(robot), name="robot", base_position=(*ROBOT_XY, RISER),
                     base_quaternion=eq._quat(ROBOT_YAW))
    floor(scene)
    eq.riser(scene, "riser", ROBOT_XY, RISER, floor_plate=1.8)
    scene.allow_link_obstacle_contact("base_link", "riser/top", robot="robot")
    totes = tote_infeed(scene, count)
    pallets = pallet_line(scene)
    scene.set_joint_positions(READY, robot="robot")
    taught = teach(scene, count, READY)
    seqs = programs(scene, totes, taught)
    for name in totes:          # the frame rests on the rim it lifts by
        scene.allow_link_obstacle_contact("eoat_mount", name, robot="robot")
    scene.set_part("robot/tool", category="gripper.mechanical", description="tote gripper, integrator-built")
    if robot == "builtin":
        scene.set_part("robot", category="robot.palletizer", model="generic 4-axis palletizer",
                       description="drawn from primitives in the class of the M-410iC/185, not a product")
    return scene, {"totes": totes, "pallets": pallets, "seqs": seqs, "layers": layers}


def bake(robot: str = "catalog", layers: int = LAYERS, max_duration: float = 600.0):
    scene, info = build(robot, layers)
    tl = scene.simulate_sequences(info["seqs"], max_duration=max_duration)
    return scene, tl, info


def report(scene: bt.Scene, tl, info: dict) -> None:
    spans = {name: (a, b) for name, a, b in tl.step_spans}
    totes = info["totes"]
    first = spans["robot/wait0"][0]
    full = spans["robot/full"][0]
    per = (full - first) / len(totes)
    print(f"tote palletizer: {tl.duration:.1f} s")
    print(f"  pallet in the station at {spans['pallets/in_station'][0]:.1f} s")
    print(f"  {len(totes)} totes, {PER_LAYER} a layer x {info['layers']} layers, "
          f"{per:.1f} s a tote ({3600 / per:.0f} totes an hour)")
    for layer in range(info["layers"]):
        done = spans[f"robot/release{(layer + 1) * PER_LAYER - 1}"][0]
        print(f"    layer {layer + 1} complete at {done:6.1f} s")
    print(f"  pallet full at {full:.1f} s, out at {spans['pallets/out'][0]:.1f} s; the next pallet "
          f"in the station at {spans['pallets/next_ready'][0]:.1f} s")
    print(f"  stack height {deck_z() + info['layers'] * (TOTE[2] + SEAT):.2f} m above the floor")
    print(f"  robot moving {100 * tl.utilizations().get('robot', 0):.0f} % of the time")


def main() -> None:
    global CATALOG_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", default=str(HERE / "tote_palletizing.usdc"))
    parser.add_argument("--studio", action="store_true")
    parser.add_argument("--robot", choices=("catalog", "builtin"), default="catalog",
                        help="the FANUC M-410iC/185 from the catalog, or a generic 4-axis palletizer (no download)")
    parser.add_argument("--layers", type=int, default=LAYERS, help=f"layers to stack (1-{LAYERS})")
    parser.add_argument("--catalog-root", type=Path, default=None)
    args = parser.parse_args()
    CATALOG_ROOT = args.catalog_root
    scene, tl, info = bake(args.robot, args.layers)
    report(scene, tl, info)
    tl.export_usd(args.out, fps=20)
    print(f"wrote {args.out}")
    if args.studio:
        bt.studio(scene, view=STUDIO_VIEW)


if __name__ == "__main__":
    main()
