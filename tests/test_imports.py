"""Import boundaries: who may import whom (see STRUCTURE.md)."""
import subprocess
import sys

from tests.helpers import APP


def test_importing_data_does_not_load_heavy_modules():
    code = (
        "import sys, cardiotriage.data;"
        "bad=[m for m in ('sklearn','xgboost','matplotlib','seaborn') if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, f"heavy modules imported by cardiotriage.data: {p.stdout.strip()} {p.stderr[-300:]}"


def test_package_init_is_empty():
    from pathlib import Path
    import cardiotriage
    assert Path(cardiotriage.__file__).read_text(encoding="utf-8").strip() == ""


def test_config_imports_nothing_from_package():
    from pathlib import Path
    import cardiotriage.config as cfg
    src = Path(cfg.__file__).read_text(encoding="utf-8")
    assert "cardiotriage" not in src and "from ." not in src, "config.py must not import package modules"


def test_importing_plots_does_not_load_ml_libraries():
    code = (
        "import sys, cardiotriage.plots;"
        "bad=[m for m in ('sklearn','xgboost','cardiotriage.models','cardiotriage.evaluation') if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, f"plots imports ML modules: {p.stdout.strip()} {p.stderr[-300:]}"


def test_features_imports_only_config_from_package():
    code = ("import sys, cardiotriage.features;"
            "bad=[m for m in sys.modules if m.startswith('cardiotriage.') and m not in ('cardiotriage.config','cardiotriage.features')];"
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr[-300:]


def test_importing_triage_does_not_load_training_code():
    code = ("import sys, cardiotriage.triage;"
            "bad=[m for m in ('cardiotriage.evaluation','cardiotriage.models','cardiotriage.plots','matplotlib') if m in sys.modules];"
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr[-300:]


def test_app_imports_only_inference_modules():
    import ast
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {node.module} | {f"{node.module}.{a.name}" for a in node.names}
    banned = {"cardiotriage.evaluation", "cardiotriage.models", "cardiotriage.plots", "matplotlib", "seaborn"}
    assert not (names & banned), names & banned
