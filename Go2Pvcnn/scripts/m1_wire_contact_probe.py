"""Deterministic selected-wire contact/lift probe using the Play startup path."""

from __future__ import annotations

import os
import sys

from scripts import m1_dual_panda_o6_bimanual_play as play


def _parser():
    parser = play._parser(wire_contact_probe=True)
    parser.set_defaults(object_assets_root=play.ROOT / "assets" / "m1_objects")
    parser.description = __doc__
    parser.epilog = "IsaacLab launcher options, including --headless, are accepted at startup."
    return parser


def main() -> int:
    return play.main(_parser())


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
