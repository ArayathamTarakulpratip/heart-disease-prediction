"""Data loading, cleaning and the data contract (cardiotriage/data.py)."""
import numpy as np
import pandas as pd


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
    data.validate(data.load_cleveland())


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


def test_kaggle_vs_uci_finds_the_dropped_and_duplicated_patient():
    # Kaggle lost one diseased UCI patient (row 184) and duplicated a healthy one: 302 unique patients, not 303.
    from cardiotriage import data
    r = data.kaggle_vs_uci()
    assert r["kaggle_duplicates"] == 1, r["kaggle_duplicates"]
    lost = r["uci_unmatched"]
    assert list(lost.index) == [184], lost
    assert int(lost["disease"].iloc[0]) == 1, lost


def test_validate_rejects_the_recoded_kaggle_file():
    # Kaggle codes (cp=0, slope=0, thal=0, ca=4...) are not valid UCI codes: the contract must catch it.
    from cardiotriage import data
    try:
        data.validate(data.load_kaggle())
    except ValueError as e:
        assert "cp" in str(e), e
    else:
        raise AssertionError("the Kaggle file must not pass the UCI data contract")


def test_dictionary_describes_all_13_features_with_official_codes():
    from cardiotriage import data
    d = data.dictionary()
    assert len(d) == 13 and set(d.index) == set(__import__("cardiotriage.config").config.FEATURES), d.index
    assert "4 = asymptomatic" in d.loc["cp", "codes"], d.loc["cp", "codes"]
    assert d["meaning"].str.len().min() > 0


def test_deduplicate_removes_exact_duplicates_only():
    from cardiotriage import data
    df = pd.concat([data.load_cleveland().head(5)] * 2, ignore_index=True)
    out = data.deduplicate(df)
    assert len(out) == 5, len(out)


def test_deduplicate_keeps_the_original_uci_row_ids():
    # the notebook names patients by UCI row (e.g. 184); dropping a duplicate must not renumber the rest
    from cardiotriage import data
    df = data.load_cleveland()
    file = pd.concat([df.iloc[:150], df.iloc[[149]], df.iloc[150:]], ignore_index=True)  # row 150 copies 149
    out = data.deduplicate(file)
    assert list(out.index) == [i for i in range(len(file)) if i != 150], list(out.index[148:153])
