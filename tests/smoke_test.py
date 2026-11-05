"""Smoke tests for the cardiotriage package. Assert-based, no pytest needed.

Run from the repo root:  python -m tests.smoke_test
Each test_* function is independent; the runner prints PASS/FAIL and exits 1 on any failure.
"""
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd


# ---------- Task 2: config + data contract ----------

def test_cleveland_loads_with_expected_shape_and_label():
    from cardiotriage import data
    df = data.load_cleveland()
    assert df.shape == (303, 15), df.shape            # 13 features + num + disease
    assert int(df["disease"].sum()) == 139, df["disease"].sum()
    assert df["ca"].isna().sum() == 4 and df["thal"].isna().sum() == 2


def test_hungarian_loads():
    from cardiotriage import data
    df = data.load_hungarian()
    assert len(df) == 294 and "disease" in df.columns


def test_validate_accepts_clean_cleveland():
    from cardiotriage import data
    data.validate(data.load_cleveland())              # must not raise


def test_validate_rejects_bad_chest_pain_code():
    from cardiotriage import data
    df = data.load_cleveland().copy()
    df.loc[0, "cp"] = 5
    try:
        data.validate(df)
    except ValueError as e:
        assert "cp" in str(e), e
    else:
        raise AssertionError("cp=5 should be rejected")


def test_validate_rejects_negative_age():
    from cardiotriage import data
    df = data.load_cleveland().copy()
    df.loc[0, "age"] = -1
    try:
        data.validate(df)
    except ValueError as e:
        assert "age" in str(e), e
    else:
        raise AssertionError("age=-1 should be rejected")


def test_validate_rejects_missing_value_outside_ca_thal():
    from cardiotriage import data
    df = data.load_cleveland().copy()
    df.loc[0, "chol"] = np.nan
    try:
        data.validate(df)
    except ValueError as e:
        assert "chol" in str(e), e
    else:
        raise AssertionError("NaN in chol should be rejected")


def test_validate_rejects_missing_column():
    from cardiotriage import data
    df = data.load_cleveland().drop(columns=["thalach"])
    try:
        data.validate(df)
    except ValueError as e:
        assert "thalach" in str(e), e
    else:
        raise AssertionError("missing column should be rejected")


def test_validate_rejects_invalid_ca_code():
    from cardiotriage import data
    df = data.load_cleveland().copy()
    df.loc[0, "ca"] = 4                                # the invalid Kaggle code (T3)
    try:
        data.validate(df)
    except ValueError as e:
        assert "ca" in str(e), e
    else:
        raise AssertionError("ca=4 should be rejected")


def test_kaggle_vs_uci_proves_label_flip_and_recoding():
    from cardiotriage import data
    r = data.kaggle_vs_uci()
    assert r["n_matched"] == r["n_kaggle"] == 303, (r["n_matched"], r["n_kaggle"])
    # target is inverted: Kaggle target=1 <-> UCI disease=0, with zero exceptions
    ct = r["crosstabs"]["target"]
    assert ct.loc[1, 0] == 165 and ct.loc[0, 1] == 138, ct
    assert ct.loc[1, 1] == 0 and ct.loc[0, 0] == 0, ct
    # every re-mapped code lines up once the documented map is applied
    assert all(v == 0 for v in r["mismatches"].values()), r["mismatches"]


def test_validate_rejects_the_recoded_kaggle_file():
    # Kaggle codes (cp=0, slope=0, thal=0, ca=4...) are not valid UCI codes: the contract must catch it.
    from cardiotriage import data
    try:
        data.validate(data.load_kaggle())
    except ValueError as e:
        assert "cp" in str(e), e
    else:
        raise AssertionError("the Kaggle file must not pass the UCI data contract")


def test_deduplicate_removes_exact_duplicates_only():
    from cardiotriage import data
    df = pd.concat([data.load_cleveland().head(5)] * 2, ignore_index=True)
    out = data.deduplicate(df)
    assert len(out) == 5, len(out)


# ---------- Checkpoint A: imports are clean ----------

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


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
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
    sys.exit(main())
