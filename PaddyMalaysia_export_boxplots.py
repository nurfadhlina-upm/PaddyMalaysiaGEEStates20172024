r"""
Create annual box-and-whisker plots from the Malaysian paddy input CSV.

Each box contains the 13 state observations for one year. The red diamond is
the mean, and its label reports mean +/- sample standard deviation.

PyCharm parameters example:
--data "C:\Users\fadhlina\PycharmProjects\PaddyGEE_Malaysia_2017-2022\input files\malaysia_paddy_enriched_data.csv" --output "Output_Malaysia_Yield_Maps\annual_variable_boxplots.png"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIABLES = [
    ("actual_yield_kg_ha", "Observed yield", r"t ha$^{-1}$", 2),
    ("Median_NDVI", "Peak NDVI", "Index", 3),
    ("Solar_Radiation_MJ", "Solar radiation", "MJ", 2),
    ("Mean_Temp_C", "Mean temperature", "°C", 2),
    ("Annual_Rain_mm", "Annual rainfall", "mm", 0),
    ("Soil_pH", "Soil pH", "pH", 2),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export annual boxplots for paddy-yield variables.")
    parser.add_argument("--data", required=True, help="Path to malaysia_paddy_enriched_data.csv")
    parser.add_argument("--output", default="annual_variable_boxplots.png", help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv("C:/Users/fadhlina/PycharmProjects/Data Paddy/malaysia_paddy_state_gee_perangkaan_2017_2024.csv")

    required = {"state", "Year", *(item[0] for item in VARIABLES)}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input file is missing columns: {sorted(missing)}")

    data["Year"] = pd.to_numeric(data["Year"], errors="raise").astype(int)
    for column, _, _, _ in VARIABLES:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if data[list(required)].isna().any().any():
        raise ValueError("Missing values were found in required columns.")
    if data.duplicated(["state", "Year"]).any():
        raise ValueError("Duplicate state-year rows were found.")

    years = sorted(data["Year"].unique())
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Also export the exact mean and SD values shown in the figure.
    summary_rows = []
    for column, title, unit, _ in VARIABLES:
        for year in years:
            values = data.loc[data["Year"].eq(year), column]
            summary_rows.append({
                "Variable": column,
                "Display_name": title,
                "Unit": unit,
                "Year": year,
                "N_states": int(values.count()),
                "Mean": values.mean(),
                "SD": values.std(ddof=1),
                "Median": values.median(),
                "Q1": values.quantile(0.25),
                "Q3": values.quantile(0.75),
                "Minimum": values.min(),
                "Maximum": values.max(),
            })
    pd.DataFrame(summary_rows).to_csv(
        output_path.with_name("annual_boxplot_values.csv"), index=False
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    box_colours = ["#dcefe3", "#cce8df", "#b9dfda", "#a8d4d5", "#91c4cf", "#78b1c5"]
    rng = np.random.default_rng(2026)

    for ax, (column, title, unit, decimals) in zip(axes.flat, VARIABLES):
        groups = [data.loc[data["Year"].eq(year), column].to_numpy() for year in years]
        result = ax.boxplot(
            groups,
            positions=np.arange(1, len(years) + 1),
            widths=0.58,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "#b43a2f", "markeredgecolor": "white", "markersize": 6},
            medianprops={"color": "#1f2f2a", "linewidth": 1.6},
            whiskerprops={"color": "#52645e", "linewidth": 1.1},
            capprops={"color": "#52645e", "linewidth": 1.1},
            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#b43a2f", "markeredgecolor": "none", "alpha": 0.7},
        )
        for patch, colour in zip(result["boxes"], box_colours):
            patch.set_facecolor(colour)
            patch.set_edgecolor("#355d4c")
            patch.set_linewidth(1.2)

        # Show the 13 state observations behind each box.
        for position, values in enumerate(groups, start=1):
            jitter = rng.normal(0, 0.045, size=len(values))
            ax.scatter(position + jitter, values, s=12, color="#355d4c", alpha=0.38, zorder=2)

        ymin = min(values.min() for values in groups)
        ymax = max(values.max() for values in groups)
        span = max(ymax - ymin, 1e-9)
        ax.set_ylim(ymin - 0.10 * span, ymax + 0.23 * span)

        # Mark the numerical mean +/- SD above each box.
        for position, values in enumerate(groups, start=1):
            mean = values.mean()
            sd = values.std(ddof=1)
            label = f"{mean:.{decimals}f}±{sd:.{decimals}f}"
            ax.text(position, ymax + 0.08 * span, label, ha="center", va="bottom",
                    fontsize=7.2, rotation=35, color="#7a2923")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(unit)
        ax.set_xticks(np.arange(1, len(years) + 1), [str(year) for year in years])
        ax.grid(axis="y", color="#d9dfdc", linewidth=0.7, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Annual distributions of paddy yield and environmental variables across 13 Malaysian states",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "Boxes show the interquartile range; centre lines show medians; whiskers extend to 1.5×IQR; "
        "points show states; red diamonds and labels show mean ± SD.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure exported to: {output_path.resolve()}")
    print(f"Values exported to: {output_path.with_name('annual_boxplot_values.csv').resolve()}")


if __name__ == "__main__":
    main()
