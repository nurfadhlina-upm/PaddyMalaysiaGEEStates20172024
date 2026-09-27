r"""
Export reproducible Malaysian state-level paddy-yield maps.

The observed map reads yield values directly from malaysia_paddy_enriched_data.csv.
If a hybrid prediction CSV is also supplied, the script exports matched observed
and predicted maps for one fixed model and ablation setting.

Required packages:
    pip install pandas numpy matplotlib

Example (observed maps only):
    python PaddyMalaysia_export_yield_maps.py ^
      --data "C:\path\input files\malaysia_paddy_enriched_data.csv"

Example (observed and predicted maps):
    python PaddyMalaysia_export_yield_maps.py ^
      --data "C:\path\input files\malaysia_paddy_enriched_data.csv" ^
      --predictions "C:\path\Output_Malaysia_HybridResidualForecast\hybrid_state_corrections.csv" ^
      --ablation "Rainfall Only" ^
      --model "Hybrid Residual OLS"
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


# Public state-boundary file used only to draw the map.
# For a final publication, replace this with the authoritative boundary file
# selected by the authors and cite that source in the manuscript.
DEFAULT_BOUNDARY_URL = (
    "https://raw.githubusercontent.com/mptwaktusolat/"
    "jakim.geojson/master/malaysia.state.geojson"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export observed and predicted paddy-yield maps.")
    parser.add_argument("--data", required=True, help="Path to malaysia_paddy_enriched_data.csv")
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional path to hybrid_state_corrections.csv",
    )
    parser.add_argument(
        "--boundaries",
        default=None,
        help="Optional local Malaysia state GeoJSON. If omitted, a public file is downloaded.",
    )
    parser.add_argument("--output", default="Output_Malaysia_Yield_Maps")
    parser.add_argument("--ablation", default="Rainfall Only")
    parser.add_argument("--model", default="Hybrid Residual OLS")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_boundaries(boundary_path: str | None, output_dir: Path) -> dict:
    """Load a local boundary file or download it once into the output folder."""
    if boundary_path:
        path = Path(boundary_path)
    else:
        path = output_dir / "malaysia_state_boundaries.geojson"
        if not path.exists():
            print("Downloading Malaysian state boundaries...")
            urllib.request.urlretrieve(DEFAULT_BOUNDARY_URL, path)

    if not path.exists():
        raise FileNotFoundError(f"Boundary file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_observed_data(data: pd.DataFrame) -> pd.DataFrame:
    """Check required columns and verify the observed-yield calculation."""
    required = {"state", "Year", "planted_area", "production", "actual_yield_t_ha"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Observed data are missing columns: {sorted(missing)}")

    data = data.copy()
    data["Year"] = pd.to_numeric(data["Year"], errors="raise").astype(int)
    numeric_columns = ["planted_area", "production", "actual_yield_t_ha"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="raise")

    if data[list(required)].isna().any().any():
        raise ValueError("Missing values were found in required observed-data columns.")
    if (data["planted_area"] <= 0).any():
        raise ValueError("planted_area must be greater than zero.")
    if data.duplicated(["state", "Year"]).any():
        raise ValueError("The observed file contains duplicate state-year rows.")

    calculated = data["production"] / data["planted_area"]
    if not np.allclose(calculated, data["actual_yield_t_ha"], rtol=1e-7, atol=1e-9):
        raise ValueError("actual_yield_t_ha does not agree with production / planted_area.")
    return data


def geometry_polygons(geometry: dict) -> list[np.ndarray]:
    """Convert GeoJSON Polygon or MultiPolygon coordinates for Matplotlib."""
    coordinates = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        coordinates = [coordinates]
    return [np.asarray(polygon[0]) for polygon in coordinates]


def draw_map(ax, boundaries: dict, values: dict, vmin: float, vmax: float, title: str):
    """Draw one state-level choropleth panel."""
    colour_map = mpl.colormaps["YlGnBu"]
    normaliser = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for feature in boundaries["features"]:
        state_name = feature["properties"]["name"]
        value = values.get(state_name, np.nan)
        colour = "#eeeeee" if np.isnan(value) else colour_map(normaliser(value))
        state_shapes = [
            Polygon(points, closed=True)
            for points in geometry_polygons(feature["geometry"])
        ]
        collection = PatchCollection(
            state_shapes,
            facecolor=colour,
            edgecolor="white",
            linewidth=0.35,
        )
        ax.add_collection(collection)

    ax.set_xlim(99.4, 119.5)
    ax.set_ylim(0.5, 7.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=3)
    return colour_map, normaliser


def export_observed_outputs(data: pd.DataFrame, boundaries: dict, output_dir: Path, dpi: int):
    """Export values, descriptive summaries and the six observed-yield maps."""
    values = data[["state", "Year", "actual_yield_t_ha"]].sort_values(["Year", "state"])
    values.to_csv(output_dir / "observed_state_year_yield_values.csv", index=False)

    annual = data.groupby("Year")["actual_yield_t_ha"].agg(
        N="count", Mean="mean", SD="std", Minimum="min", Maximum="max"
    )
    annual.to_csv(output_dir / "annual_observed_yield_summary.csv")

    years = sorted(data["Year"].unique())
    number_columns = 3
    number_rows = int(np.ceil(len(years) / number_columns))
    fig, axes = plt.subplots(number_rows, number_columns, figsize=(12, 2.8 * number_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    vmin = float(data["actual_yield_t_ha"].min())
    vmax = float(data["actual_yield_t_ha"].max())
    for ax, year in zip(axes, years):
        year_values = data.loc[data["Year"].eq(year)].set_index("state")["actual_yield_t_ha"].to_dict()
        colour_map, normaliser = draw_map(ax, boundaries, year_values, vmin, vmax, str(year))
    for ax in axes[len(years):]:
        ax.axis("off")

    scalar = mpl.cm.ScalarMappable(norm=normaliser, cmap=colour_map)
    colour_bar = fig.colorbar(scalar, ax=axes.tolist(), orientation="horizontal", fraction=0.045, pad=0.02, aspect=45)
    colour_bar.set_label("Observed paddy yield (t ha$^{-1}$)")
    fig.suptitle("Observed paddy yield across Malaysian states", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "observed_yield_maps.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def select_predictions(path: str, ablation: str, model: str) -> pd.DataFrame:
    """Select one fixed hybrid model from hybrid_state_corrections.csv."""
    predictions = pd.read_csv(path)
    required = {
        "Ablation", "Model", "Test_year", "State",
        "Actual_yield_t_ha", "Final_predicted_yield_t_ha",
        "State_yield_baseline_t_ha", "Predicted_correction_t_ha",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    selected = predictions[
        predictions["Ablation"].eq(ablation) & predictions["Model"].eq(model)
    ].copy()
    if selected.empty:
        available = predictions[["Ablation", "Model"]].drop_duplicates()
        raise ValueError(
            f"No rows matched Ablation={ablation!r} and Model={model!r}.\n"
            f"Available combinations:\n{available.to_string(index=False)}"
        )
    if selected.duplicated(["State", "Test_year"]).any():
        raise ValueError("Selected predictions contain duplicate state-test year rows.")
    return selected


def export_prediction_outputs(selected: pd.DataFrame, boundaries: dict, output_dir: Path, dpi: int):
    """Export mapped prediction values and observed-versus-predicted panels."""
    selected = selected.sort_values(["Test_year", "State"])
    selected.to_csv(output_dir / "mapped_observed_predicted_and_correction_values.csv", index=False)

    years = sorted(selected["Test_year"].astype(int).unique())
    all_yields = pd.concat([
        selected["Actual_yield_t_ha"], selected["Final_predicted_yield_t_ha"]
    ])
    vmin, vmax = float(all_yields.min()), float(all_yields.max())

    fig, axes = plt.subplots(2, len(years), figsize=(3.2 * len(years), 5.8), constrained_layout=True)
    for column, year in enumerate(years):
        year_data = selected[selected["Test_year"].eq(year)]
        observed = year_data.set_index("State")["Actual_yield_t_ha"].to_dict()
        predicted = year_data.set_index("State")["Final_predicted_yield_t_ha"].to_dict()
        colour_map, normaliser = draw_map(axes[0, column], boundaries, observed, vmin, vmax, str(year))
        draw_map(axes[1, column], boundaries, predicted, vmin, vmax, "")

    axes[0, 0].text(-0.03, 0.5, "Observed", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=11, fontweight="bold")
    axes[1, 0].text(-0.03, 0.5, "Predicted", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=11, fontweight="bold")
    scalar = mpl.cm.ScalarMappable(norm=normaliser, cmap=colour_map)
    colour_bar = fig.colorbar(scalar, ax=axes, orientation="horizontal", fraction=0.045, pad=0.02, aspect=55)
    colour_bar.set_label("Paddy yield (t ha$^{-1}$)")
    model_name = selected["Model"].iloc[0]
    ablation_name = selected["Ablation"].iloc[0]
    fig.suptitle(
        f"Observed and out-of-time predicted yield: {ablation_name}, {model_name}",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_dir / "observed_vs_predicted_yield_maps.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    observed = validate_observed_data(pd.read_csv(args.data))
    boundaries = load_boundaries(args.boundaries, output_dir)
    export_observed_outputs(observed, boundaries, output_dir, args.dpi)

    if args.predictions:
        selected = select_predictions(args.predictions, args.ablation, args.model)
        export_prediction_outputs(selected, boundaries, output_dir, args.dpi)

    print(f"Files exported to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
