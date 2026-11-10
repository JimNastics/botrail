"""The conveyor-to-conveyor transfer example: its three speed knobs.

`examples/basics/conveyor_transfer_demo.py` runs from a checkout with no
downloads, so this bakes it for real and pins what each knob is for — a
faster arm shortens the cycle, a slow infeed makes the arm wait for the
next carton, a slow outfeed makes it wait for the place spot — and that
every carton ends up off the outfeed.
"""

import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
sys.path.insert(0, str(EXAMPLES / "basics"))

import conveyor_transfer_demo as demo  # noqa: E402


def bake(boxes=3, **speeds):
    scene = demo.build_cell(boxes=boxes, **speeds)
    return scene.simulate_sequences(demo.author_sequences(scene, boxes), max_duration=300.0)


def waited(tl, step, boxes):
    return sum(tl.step_span(f"robot/{step} {n}").duration for n in range(2, boxes + 1))


def test_nominal_cell_moves_every_carton_without_waiting():
    tl = bake()
    for n in range(1, 4):
        # Every carton was set down on the outfeed, then carried away.
        x, y, _ = tl.object_pose(demo.box_name(n - 1), tl.step_span(f"robot/clear {n}").end)[0]
        assert y == pytest.approx(demo.OUT_Y, abs=1e-3) and x > demo.PLACE_X
    assert waited(tl, "await carton", 3) == pytest.approx(0.0, abs=0.02)
    assert waited(tl, "await place", 3) == pytest.approx(0.0, abs=0.02)


def test_robot_speed_shortens_the_cycle():
    assert bake(robot_speed=2.0).duration < bake().duration - 3.0


def test_slow_infeed_makes_the_arm_wait():
    assert waited(bake(infeed_speed=0.03), "await carton", 3) > 1.0


def test_slow_outfeed_makes_the_arm_wait():
    assert waited(bake(outfeed_speed=0.008), "await place", 3) > 1.0
