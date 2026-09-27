"""Paddy data analysis for PyCharm.

Install dependencies in your PyCharm terminal:
    python -m pip install pandas numpy matplotlib

Run this file. Results are saved beside it in paddy_analysis_results.
Assumption: planted_area is hectares and production_t is tonnes.
Yield here means production per planted hectare, not harvested hectare.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save charts without requiring an interactive backend.
import matplotlib.pyplot as plt


CSV_PATH = Path(r"C:\Users\fadhlina\PycharmProjects\Data Paddy\malaysia_paddy_state_gee_perangkaan_2017_2024.csv")
OUTPUT_DIR = Path(__file__).resolve().parent / "paddy_analysis_results"
ENVIRONMENT = ["Median_NDVI", "Solar_Radiation_MJ", "Mean_Temp_C",
               "Annual_Rain_mm", "Soil_pH", "Soil_Clay_Pct"]


def save_chart(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start-year", type=int, default=None,
                        help="Optional filter, e.g. --start-year 2018")
    args = parser.parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"CSV not found: {args.csv}. Edit CSV_PATH or use --csv.")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    numeric = ["year", "planted_area", "production_kg", *ENVIRONMENT,
               "actual_yield_kg_ha"]
    required = ["state", *numeric]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["state"] = df["state"].astype("string").str.strip().replace("", pd.NA)
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df[["state", "year", "planted_area", "production_kg"]].isna().any().any():
        raise ValueError("Missing state, year, planted area or production. Check source data.")
    if not np.isfinite(df[numeric].dropna().to_numpy()).all():
        raise ValueError("Non-finite numeric values found.")
    if (df["year"] % 1 != 0).any():
        raise ValueError("Years must be whole numbers.")
    df["year"] = df["year"].astype(int)
    if df.duplicated(["state", "year"]).any():
        raise ValueError("Duplicate state-year records found. Resolve them before aggregation.")
    if (df["planted_area"] <= 0).any() or (df["production_kg"] < 0).any():
        raise ValueError("Area must be positive and production must be nonnegative.")
    if args.start_year is not None:
        df = df.loc[df["year"] >= args.start_year].copy()
    if df.empty:
        raise ValueError("No rows remain after filtering.")
    df = df.sort_values(["state", "year"]).reset_index(drop=True)
    df["calculated_yield_kg_ha"] = df["production_kg"] / df["planted_area"]
    valid = df["actual_yield_kg_ha"].notna()
    looks_like_kg = valid.any() and np.allclose(
        df.loc[valid, "actual_yield_kg_ha"],
        df.loc[valid, "calculated_yield_kg_ha"] * 1000, rtol=1e-5, atol=1e-5)

    notes = [
        f"Source: {args.csv}",
        f"Records: {len(df)}; states: {df.state.nunique()}; years: {df.year.min()}-{df.year.max()}",
        "Assumption: planted_area is hectares; production_t is tonnes.",
        "Calculated yield = production_kg / planted_area (per planted hectare).",
        "Aggregate yield = total production / total planted area; not an unweighted mean.",
        "Original actual_yield_kg_ha is preserved in the prepared data.",
        "Environmental correlations are descriptive associations, not causal effects.",
        "Repeated observations of each state are not independent samples.",
        "Within-state correlations remove state averages but do not adjust for shared year effects.",
    ]
    if looks_like_kg:
        notes.append("UNIT CHECK: supplied actual_yield_kg_ha equals calculated yield x 1000; it appears to be kg/ha despite its name.")
    else:
        notes.append("UNIT CHECK: verify the supplied yield definition against source metadata.")

    coverage = df.pivot(index="state", columns="year", values="production_kg").notna()
    coverage.to_csv(output / "state_year_coverage.csv")
    if not coverage.to_numpy().all():
        notes.append("WARNING: state coverage differs across years; annual totals may not be comparable.")
if __name__ == "__main__":
    main()