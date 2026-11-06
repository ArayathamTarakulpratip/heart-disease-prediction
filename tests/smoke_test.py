"""Runs every test in tests/test_*.py. Assert-based, no pytest needed.

Run from the repo root:
    python -m tests.smoke_test              # everything
    python -m tests.smoke_test triage app   # only tests/test_triage.py and tests/test_app.py
Each test_* function is independent; the runner prints PASS/FAIL and exits 1 on any failure.
"""
import importlib
import sys
import traceback
from pathlib import Path

AREAS = sorted(p.stem.removeprefix("test_") for p in Path(__file__).parent.glob("test_*.py"))


def main(areas=()) -> int:
    unknown = set(areas) - set(AREAS)
    if unknown:
        print(f"unknown area(s) {sorted(unknown)}; choose from {AREAS}")
        return 2
    tests = []
    for area in areas or AREAS:
        mod = importlib.import_module(f"tests.test_{area}")
        tests += [(f"{area}: {n}", f) for n, f in sorted(vars(mod).items()) if n.startswith("test_") and callable(f)]
    if not tests:
        print("no tests found")
        return 1
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
