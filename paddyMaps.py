"""Run in PyCharm. Install: python -m pip install pandas numpy plotly requests"""
from pathlib import Path
import json
import webbrowser
import numpy as np
import pandas as pd
import plotly.express as px
import requests

CSV_PATH = Path(r"C:\Users\fadhlina\PycharmProjects\Data Paddy\malaysia_paddy_state_gee_perangkaan_2017_2024.csv")
OUTPUT_DIR = Path(__file__).resolve().parent / "paddy_maps"
PRODUCTION_UNIT = "t"  # Actual numbers were originally labelled tonnes. Confirm source units.
MAP_YEAR = 2024
OPEN_IN_BROWSER = True
ENV = ["Median_NDVI", "Solar_Radiation_MJ", "Mean_Temp_C", "Annual_Rain_mm", "Soil_pH", "Soil_Clay_Pct"]


def state_key(value):
    value = str(value).strip().casefold()
    return {"penang": "pulau pinang", "malacca": "melaka",
            "negri sembilan": "negeri sembilan", "trengganu": "terengganu"}.get(value, value)


def save(fig, name):
    path = OUTPUT_DIR / name
    fig.write_html(path, include_plotlyjs=True)
    print(f"Saved: {path}")
    if OPEN_IN_BROWSER:
        webbrowser.open(path.resolve().as_uri())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    production = next((c for c in ["production_t", "production_kg"] if c in df), None)
    if production is None:
        raise ValueError("Use the combined CSV containing production and planted_area, not the GEE-only CSV.")
    required = ["state", "year", "planted_area", production, *ENV]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["state"] = df["state"].astype("string").str.strip().replace("", pd.NA)
    for col in ["year", "planted_area", production, *ENV]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if df[col].isin([np.inf, -np.inf]).any():
            raise ValueError(f"Infinite values in {col}")
    if df[["state", "year", "planted_area", production]].isna().any().any():
        raise ValueError("Missing state, year, area or production.")
    if (df.year % 1 != 0).any():
        raise ValueError("Years must be whole numbers.")
    df["year"] = df.year.astype(int)
    df["state_key"] = df.state.map(state_key)
    if df.duplicated(["state_key", "year"]).any():
        raise ValueError("Duplicate state-year observations.")
    if (df.planted_area <= 0).any() or (df[production] < 0).any():
        raise ValueError("Area must be positive; production must be nonnegative.")
    if PRODUCTION_UNIT not in ["t", "kg"]:
        raise ValueError("PRODUCTION_UNIT must be t or kg.")
    df["Production (t)"] = df[production] / (1000 if PRODUCTION_UNIT == "kg" else 1)
    df["Yield (t/ha)"] = df["Production (t)"] / df.planted_area
    unit_note = f"Assumes production values are {PRODUCTION_UNIT} and area is hectares."
    print(unit_note)
    print("The production_kg header conflicts with the original tonne-labelled values; verify source metadata.")
    df.to_csv(OUTPUT_DIR / "analysis_data.csv", index=False)

    heat = df.pivot(index="state", columns="year", values="Yield (t/ha)")
    fig = px.imshow(heat, text_auto=".2f", aspect="auto", color_continuous_scale="YlGn",
                    labels={"x": "Year", "y": "State", "color": "t/ha"},
                    title="Paddy yield by state and year<br><sup>" + unit_note + "</sup>")
    fig.update_layout(height=650)
    save(fig, "01_state_year_heatmap.html")

    cols = ENV + ["Yield (t/ha)"]
    corr = df[cols].corr(min_periods=3)
    corr.to_csv(OUTPUT_DIR / "pooled_correlations.csv")
    present = df[cols].notna().astype(int)
    (present.T @ present).to_csv(OUTPUT_DIR / "correlation_sample_counts.csv")
    fig = px.imshow(corr, text_auto=".2f", range_color=[-1, 1],
                    color_continuous_scale="RdBu_r", aspect="auto",
                    title="Environmental and yield correlations<br><sup>Pooled Pearson r; associations do not establish causation.</sup>")
    fig.update_layout(height=750)
    save(fig, "02_correlation_heatmap.html")

    within = df[cols] - df.groupby("state_key")[cols].transform("mean")
    for col in cols:
        if within[col].abs().max() < 1e-10:
            within[col] = np.nan
    wcorr = within.corr(min_periods=3)
    wcorr.to_csv(OUTPUT_DIR / "within_state_correlations.csv")
    fig = px.imshow(wcorr, text_auto=".2f", range_color=[-1, 1],
                    color_continuous_scale="RdBu_r", aspect="auto",
                    title="Correlations after removing state averages<br><sup>Blank cells: no variation or insufficient data. Shared year effects remain.</sup>")
    fig.update_layout(height=750)
    save(fig, "03_within_state_correlation_heatmap.html")

    # Download Malaysia state boundaries once; subsequent runs use the cache.
    boundary_file = OUTPUT_DIR / "malaysia_states.geojson"
    if not boundary_file.exists():
        print("Downloading Malaysia state boundaries from geoBoundaries...")
        response = requests.get("https://www.geoboundaries.org/api/current/gbOpen/MYS/ADM1/", timeout=60)
        response.raise_for_status()
        metadata = response.json()
        response = requests.get(metadata.get("simplifiedGeometryGeoJSON") or metadata["gjDownloadURL"], timeout=120)
        response.raise_for_status()
        boundaries = response.json()
        if not boundaries.get("features"):
            raise ValueError("Boundary download contains no features.")
        boundary_file.write_text(json.dumps(boundaries), encoding="utf-8")
        (OUTPUT_DIR / "boundary_source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    boundaries = json.loads(boundary_file.read_text(encoding="utf-8"))
    for feature in boundaries["features"]:
        feature["properties"]["state_key"] = state_key(feature["properties"]["shapeName"])
    boundary_names = {f["properties"]["state_key"] for f in boundaries["features"]}
    unmatched = set(df.state_key) - boundary_names
    if unmatched:
        raise ValueError(f"State names do not match map boundaries: {sorted(unmatched)}. Update state_key aliases.")
    current = df.loc[df.year == MAP_YEAR].copy()
    if current.empty:
        raise ValueError(f"No observations for MAP_YEAR={MAP_YEAR}.")
    fig = px.choropleth(
        current, geojson=boundaries, locations="state_key",
        featureidkey="properties.state_key", color="Yield (t/ha)",
        hover_name="state", hover_data={"state_key": False, "Yield (t/ha)": ":.2f",
                                        "Production (t)": ":,.0f", "planted_area": ":,.0f"},
        color_continuous_scale="YlGn", projection="mercator",
        title=f"Malaysia paddy yield by state, {MAP_YEAR}<br><sup>{unit_note} Grey areas have no data.</sup>")
    fig.update_geos(fitbounds="geojson", visible=False, showland=True, landcolor="#dddddd")
    fig.update_traces(marker_line_color="#555555", marker_line_width=0.6)
    fig.add_annotation(x=0, y=-0.06, xref="paper", yref="paper", showarrow=False,
                       text="Boundaries: geoBoundaries gbOpen (CC BY 4.0). State averages; not field-level paddy locations.")
    fig.update_layout(height=700, margin={"l": 10, "r": 10, "b": 70, "t": 100})
    save(fig, "04_malaysia_yield_map.html")
    (OUTPUT_DIR / "READ_ME.txt").write_text(
        unit_note + "\nYield is per planted hectare, not harvested hectare.\n"
        "Production units require confirmation because the source header was changed.\n"
        "Pooled correlations combine between-state and within-state relationships.\n"
        "Repeated state observations are not independent. No significance tests are claimed.\n"
        "Missing environmental values are excluded pairwise; see sample counts.\n"
        "The map shows state averages, not locations of individual paddy fields.\n"
        "Boundary source: https://www.geoboundaries.org/api/current/gbOpen/MYS/ADM1/\n",
        encoding="utf-8")
    print(f"Finished. Open the HTML files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
