"""The executed-notebook checker (tests/check_notebook.py)."""


def _tiny_notebook(tmp, output_text, insight, stderr=""):
    import json
    outs = [{"output_type": "stream", "name": "stdout", "text": output_text}]
    if stderr:
        outs.append({"output_type": "stream", "name": "stderr", "text": stderr})
    nb = {"cells": [
        {"cell_type": "markdown", "source": "Hypothesis", "metadata": {}},
        {"cell_type": "code", "source": "print(1)", "execution_count": 1, "outputs": outs, "metadata": {}},
        {"cell_type": "markdown", "source": insight, "metadata": {}},
    ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = tmp / "nb.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    return p


def test_notebook_checker_accepts_printed_numbers_and_rejects_invented_ones():
    import tempfile
    from pathlib import Path
    from tests.check_notebook import check
    with tempfile.TemporaryDirectory() as d:
        ok = _tiny_notebook(Path(d), "disease rate 45.9% of 303", "**Insight:** 45.9% of 303 patients.")
        assert check(ok) == [], check(ok)
        bad = _tiny_notebook(Path(d), "disease rate 45.9%", "**Insight:** 46.0% of patients.")
        assert any("46.0" in p for p in check(bad)), check(bad)
        part = _tiny_notebook(Path(d), "value 145.9", "**Insight:** 45.9 is not a substring match.")
        assert any("45.9" in p for p in check(part)), check(part)


def test_notebook_checker_flags_warnings_on_stderr():
    import tempfile
    from pathlib import Path
    from tests.check_notebook import check
    with tempfile.TemporaryDirectory() as d:
        p = _tiny_notebook(Path(d), "ok", "plain text", stderr="FutureWarning: something changed")
        assert any("stderr" in x for x in check(p)), check(p)


def test_trap_check_lists_trap_ids_missing_from_markdown():
    import json
    import tempfile
    from pathlib import Path
    from tests.check_notebook import missing_traps
    nb = {"cells": [{"cell_type": "markdown", "source": "Fixed T1, T2 and T10; T13x is not a trap id.", "metadata": {}}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "nb.ipynb"
        p.write_text(json.dumps(nb), encoding="utf-8")
        miss = missing_traps(p)
    assert "T1" not in miss and "T10" not in miss and "T3" in miss and "T13" in miss and "T18" in miss, miss
