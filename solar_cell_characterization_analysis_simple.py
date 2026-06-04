import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Thermal experiment analysis
# Coated vs uncoated aluminum
# ============================================================
# Goal:
# Compare the temperature response of coated and uncoated
# aluminum samples in the thermal vacuum experiment.
#
# Important:
# This code does NOT calculate absolute emissivity.
# It gives a relative comparison:
# the sample with lower final temperature and smaller
# sample-wall temperature difference is interpreted as having
# stronger radiative heat exchange.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UNCOATED_FILE = BASE_DIR / "uncoated_aluminum.csv"
COATED_FILE = BASE_DIR / "coated_aluminum.csv"

TAIL_WINDOW_MIN = 30
SMOOTH_WINDOW_POINTS = 25


def load_case(file_path, case_name):
    # The CSV files are tab-separated, not comma-separated
    df = pd.read_csv(file_path, sep="\t")

    # Remove extra spaces from column names
    df.columns = [col.strip() for col in df.columns]

    # Convert columns to correct data types
    # errors="coerce" turns invalid values such as "NA" into NaN
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Wall Temp"] = pd.to_numeric(df["Wall Temp"], errors="coerce")
    df["Sample Temp"] = pd.to_numeric(df["Sample Temp"], errors="coerce")

    # Remove rows with missing or invalid values
    df = df.dropna(subset=["Time", "Wall Temp", "Sample Temp"])

    # Sort by time to avoid problems if data order is not perfect
    df = df.sort_values("Time").reset_index(drop=True)

    # Create relative time axis in minutes
    df["Time_min"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds() / 60.0

    # Temperature difference between sample and wall
    # This is useful for checking thermal coupling
    df["Delta_T"] = df["Sample Temp"] - df["Wall Temp"]

    # Smooth the curves to reduce measurement noise
    df["Wall_Smooth"] = df["Wall Temp"].rolling(
        SMOOTH_WINDOW_POINTS,
        center=True,
        min_periods=1
    ).mean()

    df["Sample_Smooth"] = df["Sample Temp"].rolling(
        SMOOTH_WINDOW_POINTS,
        center=True,
        min_periods=1
    ).mean()

    df["Delta_T_Smooth"] = df["Sample_Smooth"] - df["Wall_Smooth"]

    df["Case"] = case_name

    return df


def final_window(df, minutes):
    # Select the last part of the experiment
    # This is used as approximate steady-state behavior
    end_time = df["Time_min"].max()
    return df[df["Time_min"] >= end_time - minutes]


def heating_rate(df, minutes=60):
    # Estimate early heating rate using a linear fit
    early = df[df["Time_min"] <= minutes]

    if len(early) < 2:
        return np.nan

    x = early["Time_min"].to_numpy()
    y = early["Sample_Smooth"].to_numpy()

    slope_k_per_min = np.polyfit(x, y, 1)[0]

    # Convert from K/min to K/hour
    return slope_k_per_min * 60.0


def get_metrics(df):
    # Compute compact values for the results table
    tail = final_window(df, TAIL_WINDOW_MIN)

    return {
        "duration_min": df["Time_min"].max(),
        "sample_start_K": df["Sample Temp"].iloc[0],
        "sample_end_K": df["Sample Temp"].iloc[-1],
        "wall_start_K": df["Wall Temp"].iloc[0],
        "wall_end_K": df["Wall Temp"].iloc[-1],
        "sample_tail_mean_K": tail["Sample Temp"].mean(),
        "wall_tail_mean_K": tail["Wall Temp"].mean(),
        "delta_T_tail_mean_K": tail["Delta_T"].mean(),
        "delta_T_tail_std_K": tail["Delta_T"].std(),
        "heating_rate_first_60_min_K_per_h": heating_rate(df),
        "max_sample_temp_K": df["Sample Temp"].max(),
        "max_wall_temp_K": df["Wall Temp"].max(),
    }


def print_metrics(name, metrics):
    print(f"\n=== {name.upper()} SAMPLE ===")
    for key, value in metrics.items():
        print(f"{key:35s}: {value:.4f}")


def plot_temperature_history(uncoated, coated):
    # Main plot for the report:
    # compares sample and wall temperature history
    plt.figure(figsize=(10, 5))

    plt.plot(uncoated["Time_min"], uncoated["Sample_Smooth"], label="Uncoated sample")
    plt.plot(uncoated["Time_min"], uncoated["Wall_Smooth"], "--", label="Uncoated wall")

    plt.plot(coated["Time_min"], coated["Sample_Smooth"], label="Coated sample")
    plt.plot(coated["Time_min"], coated["Wall_Smooth"], "--", label="Coated wall")

    plt.xlabel("Time [min]")
    plt.ylabel("Temperature [K]")
    plt.title("Temperature History")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(BASE_DIR / "temperature_history.png", dpi=300)
    plt.show()


def plot_delta_t(uncoated, coated):
    # Second important plot:
    # compares sample-wall temperature difference
    plt.figure(figsize=(10, 5))

    plt.plot(
        uncoated["Time_min"],
        uncoated["Delta_T_Smooth"],
        label="Uncoated: sample - wall",
    )

    plt.plot(
        coated["Time_min"],
        coated["Delta_T_Smooth"],
        label="Coated: sample - wall",
    )

    plt.axhline(0, color="black", linewidth=0.8)

    plt.xlabel("Time [min]")
    plt.ylabel(r"$\Delta T = T_{sample} - T_{wall}$ [K]")
    plt.title("Sample-Wall Temperature Difference")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(BASE_DIR / "delta_T_sample_wall.png", dpi=300)
    plt.show()


def make_summary_table(uncoated_metrics, coated_metrics):
    # Small table for the report
    table = pd.DataFrame(
        {
            "Metric": [
                "Duration [min]",
                "Final sample temperature [K]",
                f"Mean sample temperature, final {TAIL_WINDOW_MIN} min [K]",
                f"Mean wall temperature, final {TAIL_WINDOW_MIN} min [K]",
                f"Mean Delta T, final {TAIL_WINDOW_MIN} min [K]",
                "Heating rate, first 60 min [K/h]",
                "Maximum sample temperature [K]",
            ],
            "Uncoated": [
                uncoated_metrics["duration_min"],
                uncoated_metrics["sample_end_K"],
                uncoated_metrics["sample_tail_mean_K"],
                uncoated_metrics["wall_tail_mean_K"],
                uncoated_metrics["delta_T_tail_mean_K"],
                uncoated_metrics["heating_rate_first_60_min_K_per_h"],
                uncoated_metrics["max_sample_temp_K"],
            ],
            "Coated": [
                coated_metrics["duration_min"],
                coated_metrics["sample_end_K"],
                coated_metrics["sample_tail_mean_K"],
                coated_metrics["wall_tail_mean_K"],
                coated_metrics["delta_T_tail_mean_K"],
                coated_metrics["heating_rate_first_60_min_K_per_h"],
                coated_metrics["max_sample_temp_K"],
            ],
        }
    )

    table["Difference Coated - Uncoated"] = table["Coated"] - table["Uncoated"]

    print("\n=== SUMMARY TABLE ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    table.to_csv(BASE_DIR / "thermal_summary_table.csv", index=False)

    # LaTeX table file, ready to copy into Overleaf
    latex_table = table.to_latex(index=False, float_format="%.3f")
    with open(BASE_DIR / "thermal_summary_table.tex", "w") as f:
        f.write(latex_table)


def compare_emissivity(uncoated_metrics, coated_metrics):
    # Relative interpretation only
    uncoated_sample = uncoated_metrics["sample_tail_mean_K"]
    coated_sample = coated_metrics["sample_tail_mean_K"]

    uncoated_gap = abs(uncoated_metrics["delta_T_tail_mean_K"])
    coated_gap = abs(coated_metrics["delta_T_tail_mean_K"])

    print("\n=== RELATIVE EMISSIVITY INTERPRETATION ===")

    print(f"Uncoated final mean sample temperature: {uncoated_sample:.4f} K")
    print(f"Coated final mean sample temperature:   {coated_sample:.4f} K")
    print(f"Uncoated final mean |Delta T|:          {uncoated_gap:.4f} K")
    print(f"Coated final mean |Delta T|:            {coated_gap:.4f} K")

    if coated_sample < uncoated_sample and coated_gap < uncoated_gap:
        print("Conclusion: the coated sample shows stronger radiative heat exchange.")
        print("Interpretation: the coated sample has higher effective emissivity.")
    elif coated_sample < uncoated_sample:
        print("Conclusion: the coated sample reached a lower final temperature.")
        print("Interpretation: this indicates higher effective emissivity.")
    else:
        print("Conclusion: the expected trend is not fully clear from this dataset.")
        print("Interpretation: check sensor placement, heating conditions, and data quality.")


def main():
    print("Loading data files...")
    print(f"Uncoated file: {UNCOATED_FILE}")
    print(f"Coated file:   {COATED_FILE}")

    uncoated = load_case(UNCOATED_FILE, "uncoated")
    coated = load_case(COATED_FILE, "coated")

    uncoated_metrics = get_metrics(uncoated)
    coated_metrics = get_metrics(coated)

    print_metrics("uncoated", uncoated_metrics)
    print_metrics("coated", coated_metrics)

    make_summary_table(uncoated_metrics, coated_metrics)
    compare_emissivity(uncoated_metrics, coated_metrics)

    plot_temperature_history(uncoated, coated)
    plot_delta_t(uncoated, coated)

    print("\nSaved output files:")
    print("temperature_history.png")
    print("delta_T_sample_wall.png")
    print("thermal_summary_table.csv")
    print("thermal_summary_table.tex")


if __name__ == "__main__":
    main()
