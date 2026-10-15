"""
check_consistency.py
====================
Final consistency checks for the project. Exits non-zero, listing every failed check, if:

- a number in a generated README table differs from the file it was generated from,
- README.md or analysis.ipynb contains an absolute path (drive letter, file:///, /Users/),
- banned wording appears in README.md, analysis.ipynb, run_pipeline.py, or src/,
- train rows plus test rows differ from the processed parquet row count,
- a test row is dated before the split date in src/models.py,
- the saved headline XGBoost model uses any entry of SNAPSHOT_FEATURES,
- a cell in analysis.ipynb has an error output,
- a code cell in analysis.ipynb references one of the raw CSVs in the project root.

Usage (from the project root):
    python tests/check_consistency.py
"""

import ast
import contextlib
import io
import json
import os
import re
import sys

import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.check_readme import ABSOLUTE_PATH_RE, check as check_readme_tables, figures, marker_block, table_rows

README = "README.md"
NOTEBOOK = "analysis.ipynb"
PARQUET = "data/processed_market_valuations.parquet"
PREDICTIONS = "models/test_predictions.parquet"
XGB_MODEL = "models/xgboost_market_value_model.json"
MODELS_SRC = "src/models.py"

# Built from parts so this file does not contain the banned strings itself.
BANNED_STRINGS = ["leak" + "-free", "Code Guardrail " + "Verified"]
RAW_CSVS = ["player_valuations.csv", "players.csv", "appearances.csv", "clubs.csv", "competitions.csv", "games.csv"]


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def models_constants():
    """SNAPSHOT_FEATURES and the split date, parsed from src/models.py without importing it."""
    source = read(MODELS_SRC)
    snapshot = None
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "SNAPSHOT_FEATURES"):
            snapshot = ast.literal_eval(node.value)
    split = re.search(r'split_date\s*=\s*"(\d{4}-\d{2}-\d{2})"', source)
    if snapshot is None or split is None:
        raise SystemExit(f"Could not find SNAPSHOT_FEATURES or split_date in {MODELS_SRC}.")
    return snapshot, split.group(1)


def check_readme_numbers(readme_text, problems):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = check_readme_tables(README, "models")
    table_problems = [line.strip()[2:] for line in output.getvalue().splitlines()
                      if line.startswith("  - ") and "Absolute path" not in line]
    if status != 0 and table_problems:
        problems.append("README tables differ from the generated CSVs (src/check_readme.py):\n      - "
                        + "\n      - ".join(table_problems))

    expected = duckdb.sql(
        f"""SELECT minutes_last_365d = 0, count(*), count(*) * 100.0 / sum(count(*)) OVER (),
                   avg(abs_residual_eur), avg(abs_residual_log)
            FROM '{PREDICTIONS}' GROUP BY 1 ORDER BY 1 DESC"""
    ).fetchall()
    rows = table_rows(marker_block(readme_text, "zero_minutes_errors"))
    if len(rows) != len(expected):
        problems.append(f"README zero_minutes_errors table has {len(rows)} rows; {PREDICTIONS} gives {len(expected)}.")
        return
    for row, (_, n, share, euro_mae, log_mae) in zip(rows, expected):
        cells = [c.strip() for c in row.strip("|").split("|")]
        euros = [(v, tol) for v, tol, text in figures(row) if "€" in text]
        pcts = [(v, tol) for v, tol, text in figures(row) if "€" not in text]
        ok = (
            int(cells[1].replace(",", "")) == n
            and len(euros) == 1 and len(pcts) == 1
            and abs(pcts[0][0] - share) <= pcts[0][1]
            and abs(euros[0][0] - euro_mae) <= euros[0][1]
            and abs(float(cells[4]) - log_mae) <= 0.5e-4 + 1e-12
        )
        if not ok:
            problems.append(f"README zero_minutes_errors row '{cells[0]}' does not match {PREDICTIONS} "
                            f"(expected {n:,} rows, {share:.1f}%, €{euro_mae:,.0f}, {log_mae:.4f}).")


def check_absolute_paths(problems):
    for path in (README, NOTEBOOK):
        for lineno, line in enumerate(read(path).splitlines(), start=1):
            if ABSOLUTE_PATH_RE.search(line):
                problems.append(f"Absolute path in {path} line {lineno}: {line.strip()[:120]}")


def check_banned_strings(problems):
    paths = [README, NOTEBOOK, "run_pipeline.py"] + sorted(
        os.path.join(d, f) for d, _, files in os.walk("src") if "__pycache__" not in d for f in files if f.endswith(".py")
    )
    for path in paths:
        text = read(path)
        for banned in BANNED_STRINGS:
            if banned.lower() in text.lower():
                problems.append(f"Banned wording '{banned}' found in {path}.")


def check_split(split_date, problems):
    n_parquet, n_train = duckdb.sql(
        f"SELECT count(*), count(*) FILTER (WHERE valuation_date < DATE '{split_date}') FROM '{PARQUET}'"
    ).fetchone()
    n_test, first_test = duckdb.sql(f"SELECT count(*), min(valuation_date) FROM '{PREDICTIONS}'").fetchone()
    if n_train + n_test != n_parquet:
        problems.append(f"Train rows ({n_train:,}) + test rows ({n_test:,}) = {n_train + n_test:,}, "
                        f"but {PARQUET} has {n_parquet:,} rows.")
    n_early = duckdb.sql(
        f"SELECT count(*) FROM '{PREDICTIONS}' WHERE valuation_date < DATE '{split_date}'"
    ).fetchone()[0]
    if n_early:
        problems.append(f"{n_early:,} test rows in {PREDICTIONS} are dated before {split_date} "
                        f"(earliest: {first_test:%Y-%m-%d}).")


def check_headline_features(snapshot_features, problems):
    with open(XGB_MODEL, encoding="utf-8") as f:
        feature_names = json.load(f)["learner"]["feature_names"]
    leaked = sorted(set(feature_names) & set(snapshot_features))
    if leaked:
        problems.append(f"Headline XGBoost model ({XGB_MODEL}) uses snapshot features: {', '.join(leaked)}.")


def check_notebook(problems):
    notebook = json.loads(read(NOTEBOOK))
    raw_csv_re = re.compile(r"(?<![\w.])(" + "|".join(re.escape(name) for name in RAW_CSVS) + r")")
    for index, cell in enumerate(notebook["cells"], start=1):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                problems.append(f"{NOTEBOOK} cell {index} has an error output: "
                                f"{output.get('ename')}: {output.get('evalue')}")
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            found = sorted(set(raw_csv_re.findall(source)))
            if found:
                problems.append(f"{NOTEBOOK} code cell {index} references raw CSV(s): {', '.join(found)}.")


def main():
    os.chdir(ROOT)
    snapshot_features, split_date = models_constants()
    problems = []
    checks = [
        ("README tables match generated files", lambda: check_readme_numbers(read(README), problems)),
        ("No absolute paths in README or notebook", lambda: check_absolute_paths(problems)),
        ("No banned wording in README, notebook, run_pipeline.py, or src/", lambda: check_banned_strings(problems)),
        (f"Train + test rows equal the parquet; no test row before {split_date}", lambda: check_split(split_date, problems)),
        ("Headline XGBoost model has no SNAPSHOT_FEATURES", lambda: check_headline_features(snapshot_features, problems)),
        ("Notebook has no errors and no raw CSV references", lambda: check_notebook(problems)),
    ]
    for name, run in checks:
        before = len(problems)
        run()
        print(f"[{'PASS' if len(problems) == before else 'FAIL'}] {name}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll consistency checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
