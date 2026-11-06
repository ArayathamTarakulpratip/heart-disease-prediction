"""Check an EXECUTED notebook is clean and its insights are honest.

Run from the repo root after a headless run:
    python -m tests.check_notebook [path/to/notebook.ipynb]

Rules (SPEC §10, todo "headless run is clean"):
  1. every code cell was executed; no error outputs; no stderr text mentioning Warning/Error/Traceback
  2. every markdown cell that starts with **Insight:** only quotes numbers that appear in the outputs
     of the code cells between the previous markdown cell and this one (no invented numbers)
"""
import json
import re
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / "heart_disease_pitch_gold_standard.ipynb"
BAD = re.compile(r"Warning|Error|Traceback")
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _text(output) -> str:
    if output.get("output_type") == "stream":
        return "".join(output.get("text", ""))
    data = output.get("data", {})
    return "".join(data.get("text/plain", "")) + "".join(data.get("text/html", ""))


def _source(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def check(path: Path) -> list[str]:
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    problems, recent_output = [], ""
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            if not _source(cell).strip():
                problems.append(f"cell {i}: empty code cell")
                continue
            if cell.get("execution_count") is None:
                problems.append(f"cell {i}: not executed")
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    problems.append(f"cell {i}: error {out.get('ename')}: {out.get('evalue')}")
                if out.get("output_type") == "stream" and out.get("name") == "stderr" and BAD.search(_text(out)):
                    problems.append(f"cell {i}: stderr: {_text(out).strip()[:200]}")
                recent_output += _text(out) + "\n"
        else:
            src = _source(cell).strip()
            if src.startswith("**Insight:**"):
                for num in NUMBER.findall(src):
                    if not re.search(rf"(?<![\d.]){re.escape(num)}(?![\d])", recent_output):
                        problems.append(f"cell {i}: Insight quotes {num}, not printed by the cells above")
            recent_output = ""          # each Insight may only use the outputs since the previous markdown cell
    return problems


TRAPS = [f"T{i}" for i in range(1, 19)]


def missing_traps(path: Path) -> list[str]:
    """Trap ids (T1..T18, SPEC §2b) that no markdown cell mentions as a whole word."""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    text = "\n".join(_source(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    return [t for t in TRAPS if not re.search(rf"\b{t}\b", text)]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"FAIL  notebook not found: {path}")
        return 1
    problems = check(path) + [f"trap {t} is never referenced in the markdown" for t in missing_traps(path)]
    for p in problems:
        print(f"FAIL  {p}")
    print("notebook check: " + ("clean" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
