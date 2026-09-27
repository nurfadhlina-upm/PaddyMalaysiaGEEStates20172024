from pathlib import Path
import pandas as pd

# ---------- SETTINGS ----------
CSV_PATH = Path(
    r"C:\Users\fadhlina\PycharmProjects\Data Paddy"
    r"\malaysia_paddy_state_gee_perangkaan_2017_2024.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "paddy_analysis_results"

PRODUCTION_UNIT = "t"   # Actual unit of the numbers: "t" or "kg"
START_YEAR = None       # Use 2018 to exclude 2017, or None for all years

ENVIRONMENT = [
    "Median_NDVI",
    "Solar_Radiation_MJ",
    "Mean_Temp_C",
    "Annual_Rain_mm",
    "Soil_pH",
    "Soil_Clay_Pct",
]


def main():
    # ---------- LOAD DATA ----------
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Cannot find CSV:\n{CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    production_column = next(
        (c for c in ["production_t", "production_kg"] if c in df.columns),
        None,
    )

    if production_column is None:
        raise ValueError("CSV needs production_t or production_kg.")

    required = ["state", "year", "planted_area", production_column]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    environmental_columns = [
        c for c in ENVIRONMENT if c in df.columns
    ]

    df["state"] = (
        df["state"].astype("string").str.strip().replace("", pd.NA)
    )

    numeric_columns = [
        "year", "planted_area", production_column,
        *environmental_columns,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df[required].isna().any().any():
        raise ValueError(
            "Missing state, year, area or production. "
            "Please check the source CSV."
        )

    if df[numeric_columns].isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("Infinite numeric values found.")

    if (df["year"] % 1 != 0).any():
        raise ValueError("Year must be a whole number.")

    df["year"] = df["year"].astype(int)

    if df.duplicated(["state", "year"]).any():
        raise ValueError("Duplicate state-year records found.")

    if (df["planted_area"] <= 0).any():
        raise ValueError("Planted area must be greater than zero.")

    if (df[production_column] < 0).any():
        raise ValueError("Production cannot be negative.")

    if START_YEAR is not None:
        df = df.loc[df["year"] >= START_YEAR].copy()

    if df.empty:
        raise ValueError("No data remains after filtering.")

    df = df.sort_values(["state", "year"]).reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- STANDARDISE UNITS ----------
    # Assumes planted_area is in hectares.
    # Keep original columns and create separate calculated columns.
    if PRODUCTION_UNIT == "t":
        df["analysis_production_t"] = df[production_column]
    elif PRODUCTION_UNIT == "kg":
        df["analysis_production_t"] = df[production_column] / 1000
    else:
        raise ValueError('PRODUCTION_UNIT must be "t" or "kg".')

    df["calculated_yield_t_ha"] = (
        df["analysis_production_t"] / df["planted_area"]
    )

    df["calculated_yield_kg_ha"] = (
        df["calculated_yield_t_ha"] * 1000
    )

    # ---------- 1. MISSING VALUES ----------
    missing_values = df.isna().sum().to_frame("missing_count")
    missing_values["missing_percent"] = (
        missing_values["missing_count"] / len(df) * 100
    )
    missing_values.to_csv(OUTPUT_DIR / "01_missing_values.csv")

    # ---------- 2. STATE-YEAR COVERAGE ----------
    coverage = df.pivot(
        index="state",
        columns="year",
        values="analysis_production_t",
    ).notna()

    coverage.to_csv(OUTPUT_DIR / "02_state_year_coverage.csv")

    # ---------- 3. DESCRIPTIVE STATISTICS ----------
    analysis_columns = [
        "planted_area",
        "analysis_production_t",
        "calculated_yield_t_ha",
        *environmental_columns,
    ]

    statistics = df[analysis_columns].describe().T
    statistics.to_csv(OUTPUT_DIR / "03_descriptive_statistics.csv")

    # ---------- 4. ANNUAL SUMMARY ----------
    annual = df.groupby("year").agg(
        states_reported=("state", "nunique"),
        total_planted_area_ha=("planted_area", "sum"),
        total_production_t=("analysis_production_t", "sum"),
    )

    # Correct aggregate yield: total production / total area.
    annual["weighted_yield_t_ha"] = (
        annual["total_production_t"]
        / annual["total_planted_area_ha"]
    )

    consecutive_years = annual.index.to_series().diff().eq(1)

    for column in [
        "total_planted_area_ha",
        "total_production_t",
        "weighted_yield_t_ha",
    ]:
        previous = annual[column].shift(1)

        annual[column + "_change_pct"] = (
            (annual[column] / previous - 1) * 100
        ).where(consecutive_years & previous.ne(0))

    annual.to_csv(OUTPUT_DIR / "04_annual_summary.csv")

    # ---------- 5. STATE SUMMARY ----------
    states = df.groupby("state").agg(
        years_reported=("year", "nunique"),
        total_production_t=("analysis_production_t", "sum"),
        planted_area_ha_years=("planted_area", "sum"),
        mean_annual_production_t=("analysis_production_t", "mean"),
        mean_annual_yield_t_ha=("calculated_yield_t_ha", "mean"),
        minimum_yield_t_ha=("calculated_yield_t_ha", "min"),
        maximum_yield_t_ha=("calculated_yield_t_ha", "max"),
        yield_std_dev=("calculated_yield_t_ha", "std"),
    )

    states["weighted_yield_t_ha"] = (
        states["total_production_t"]
        / states["planted_area_ha_years"]
    )

    states["yield_rank"] = (
        states["weighted_yield_t_ha"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    states = states.sort_values("yield_rank")
    states.to_csv(OUTPUT_DIR / "05_state_summary.csv")

    # ---------- 6. YIELD BY STATE AND YEAR ----------
    yield_table = df.pivot(
        index="state",
        columns="year",
        values="calculated_yield_t_ha",
    )

    yield_table.to_csv(OUTPUT_DIR / "06_state_year_yield.csv")

    # ---------- 7. ENVIRONMENTAL CORRELATIONS ----------
    correlation_columns = [
        *environmental_columns,
        "calculated_yield_t_ha",
    ]

    pooled = df[correlation_columns].corr(
        method="pearson",
        min_periods=3,
    )

    pooled.to_csv(OUTPUT_DIR / "07_correlation_matrix.csv")

    # Remove each state's average to examine within-state variation.
    within_data = (
        df[correlation_columns]
        - df.groupby("state")[correlation_columns].transform("mean")
    )

    # A variable constant within every state has no within-state correlation.
    for column in correlation_columns:
        if within_data[column].abs().max() < 1e-10:
            within_data[column] = float("nan")

    within = within_data.corr(min_periods=3)

    relationships = pd.DataFrame({
        "pooled_correlation_with_yield": pooled.loc[
            environmental_columns, "calculated_yield_t_ha"
        ],
        "within_state_correlation_with_yield": within.loc[
            environmental_columns, "calculated_yield_t_ha"
        ],
        "paired_observations": [
            df[[column, "calculated_yield_t_ha"]].dropna().shape[0]
            for column in environmental_columns
        ],
    })

    relationships.to_csv(
        OUTPUT_DIR / "08_environment_yield_correlations.csv"
    )

    # ---------- 8. PREPARED DATA ----------
    df.to_csv(OUTPUT_DIR / "09_prepared_data.csv", index=False)

    # ---------- 9. TEXT REPORT ----------
    latest_year = int(df["year"].max())
    latest = annual.loc[latest_year]
    top_state = states.index[0]

    report = [
        "PADDY DATA ANALYSIS",
        "",
        f"Records analysed: {len(df)}",
        f"States: {df['state'].nunique()}",
        f"Period: {df['year'].min()}–{latest_year}",
        f"Production source column: {production_column}",
        f"Production values interpreted as: {PRODUCTION_UNIT}",
        "Planted area assumed to be hectares.",
        "",
        f"LATEST YEAR: {latest_year}",
        f"Production: {latest['total_production_t']:,.2f} tonnes",
        f"Planted area: {latest['total_planted_area_ha']:,.2f} ha",
        f"Weighted yield: {latest['weighted_yield_t_ha']:.3f} t/ha",
        "",
        f"Highest weighted yield over the analysed period: {top_state}",
        f"Yield: {states.loc[top_state, 'weighted_yield_t_ha']:.3f} t/ha",
        "",
        "INTERPRETATION",
        "Yield = production / planted area, not harvested area.",
        "Weighted yield = total production / total planted area.",
        "Area summed across years is hectare-years, not unique land area.",
        "Annual totals cover only the states present in the CSV.",
        "Correlations describe associations; they do not prove causation.",
        "Within-state correlations remove state averages, not year effects.",
        "Blank correlations mean insufficient data or no variation.",
        "Missing environmental observations are excluded pairwise.",
        "Repeated state observations are not independent samples.",
    ]

    if production_column == "production_kg" and PRODUCTION_UNIT == "t":
        report.append(
            "UNIT WARNING: production_kg is interpreted as tonnes based "
            "on the original file and unchanged values. Confirm the "
            "actual unit against source metadata."
        )

    if not coverage.to_numpy().all():
        report.append(
            "COVERAGE WARNING: some state-year records are missing. "
            "Annual totals may not be directly comparable."
        )

    report_text = "\n".join(report)
    (OUTPUT_DIR / "10_analysis_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    print(report_text)
    print("\nANNUAL SUMMARY")
    print(annual.round(3).to_string())
    print("\nSTATE SUMMARY")
    print(states.round(3).to_string())
    print(f"\nAll 10 output files saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()