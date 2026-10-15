"""
check_readme.py
===============
Checks that every euro and percentage figure in the generated README tables matches its source CSV.

For each table row, all euro and percentage figures must be found (within display rounding) in a
single row of the corresponding CSV. Also fails if README.md contains an absolute path or link.

Usage:
    python src/check_readme.py
"""

import argparse
import csv
import os
import re
import sys

TABLE_SOURCES = {
    "benchmark": "model_comparison_metrics.csv",
    "leakage_ablation": "leakage_ablation.csv",
    "residuals_age": "residual_analysis_by_age.csv",
    "residuals_position": "residual_analysis_by_position.csv",
    "residuals_tier": "residual_analysis_by_tier.csv",
    "top20_model_above": "undervalued_top20.csv",
    "top20_model_below": "overvalued_top20.csv",
}

EURO_RE = re.compile(r"([+-]?)€([\d,]+(?:\.\d+)?)")
PCT_RE = re.compile(r"([+-]?\d+(?:\.(\d+))?)%")
ABSOLUTE_PATH_RE = re.compile(r"file:///|(?<![A-Za-z])[A-Za-z]:[\\/]|(?<![\w.])/(?:Users|home)/")


def marker_block(text, name):
    match = re.search(rf"<!-- BEGIN:{name} -->\n(.*?)\n[ \t]*<!-- END:{name} -->", text, re.S)
    if not match:
        raise SystemExit(f"Marker pair for '{name}' not found in README.md.")
    return match.group(1)


def table_rows(block):
    lines = [l for l in block.splitlines() if l.startswith("|")]
    return [l for l in lines[2:]]  # skip header and alignment rows


def figures(row):
    """(value, tolerance, text) for every euro and percentage figure in a markdown table row."""
    found = []
    for sign, digits in EURO_RE.findall(row):
        value = float(digits.replace(",", "")) * (-1 if sign == "-" else 1)
        found.append((value, 0.5 + 1e-6, f"{sign}€{digits}"))
    row_without_euros = EURO_RE.sub("", row)
    for number, decimals in PCT_RE.findall(row_without_euros):
        found.append((float(number), 0.5 * 10 ** -len(decimals) + 1e-9, f"{number}%"))
    return found


def csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for record in csv.DictReader(f):
            values = []
            for cell in record.values():
                try:
                    values.append(float(cell.rstrip("%").replace(",", "")))
                except (ValueError, AttributeError):
                    pass
            rows.append(values)
        return rows


def check(readme_path, models_dir):
    with open(readme_path, encoding="utf-8") as f:
        text = f.read()

    problems, n_rows, n_figures = [], 0, 0
    for name, source in TABLE_SOURCES.items():
        source_rows = csv_rows(os.path.join(models_dir, source))
        readme_rows = table_rows(marker_block(text, name))
        if len(readme_rows) != len(source_rows):
            problems.append(f"[{name}] README table has {len(readme_rows)} rows, {source} has {len(source_rows)}.")
        for row in readme_rows:
            row_figures = figures(row)
            n_rows += 1
            n_figures += len(row_figures)
            matches = [
                values for values in source_rows
                if all(any(abs(v - value) <= tol for v in values) for value, tol, _ in row_figures)
            ]
            if not matches:
                missing = [t for value, tol, t in row_figures
                           if not any(abs(v - value) <= tol for values in source_rows for v in values)]
                detail = f"not in {source}: {', '.join(missing)}" if missing else f"no single row of {source} has all of them"
                problems.append(f"[{name}] {row.split('|')[1].strip()}: {detail}")

    for lineno, line in enumerate(text.splitlines(), start=1):
        if ABSOLUTE_PATH_RE.search(line):
            problems.append(f"Absolute path on README line {lineno}: {line.strip()}")

    print(f"Checked {n_figures} euro/percentage figures in {n_rows} table rows across {len(TABLE_SOURCES)} tables.")
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("All figures match their source CSVs; no absolute paths found.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check README table figures against pipeline CSVs.")
    parser.add_argument("--readme", type=str, default="README.md", help="Path to README.md.")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory with model outputs.")
    args = parser.parse_args()
    sys.exit(check(args.readme, args.models_dir))
