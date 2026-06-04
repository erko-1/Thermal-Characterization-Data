import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Thermal comparison of coated vs uncoated aluminum samples
# ------------------------------------------------------------
# What this script does:
# 1) Loads both CSV files
# 2) Cleans and aligns the data
# 3) Plots temperature histories
# 4) Computes simple comparison metrics
# 5) Gives a relative emissivity conclusion
#
# Important:
# With only temperature data, you cannot compute absolute emissivity.
# You can compare emissivity relatively: the sample that stays cooler
# for the same heating conditions is the more emissive one.
# ------------------------------------------------------------

UNCOATED_FILE = "uncoted_aluminum.csv"
COATED_FILE = "coted_aluminum.csv"

TAIL_WINDOW_MIN = 30      # steady-state window at the end
SMOOTH_WINDOW_POINTS = 25  # rolling average for cleaner plots

def load_case(path: str, case_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]

    # Parse and clean
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Wall Temp"] = pd.to_numeric(df["Wall Temp"], errors="coerce")
    df["Sample Temp"] = pd.to_numeric(df["Sample Temp"], errors="coerce")
    df = df.dropna(subset=["Time", "Wall Temp", "Sample Temp"]).sort_values("Time").reset_index(drop=True)

    # Relative time in minutes
    df["t_min"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds() / 60.0

    # Useful derived quantity
    df["gap"] = df["Sample Temp"] - df["Wall Temp"]

    # Smoothed signals for cleaner plots and more stable comparisons
    df["Wall Temp Smooth"] = df["Wall Temp"].rolling(SMOOTH_WINDOW_POINTS, center=True, min_periods=1).mean()
    df["Sample Temp Smooth"] = df["Sample Temp"].rolling(SMOOTH_WINDOW_POINTS, center=True, min_periods=1).mean()
    df["gap_smooth"] = df["Sample Temp Smooth"] - df["Wall Temp Smooth"]

    df["case"] = case_name
    return df

def last_window(df: pd.DataFrame, window_min: float) -> pd.DataFrame:
    t_end = df["t_min"].max()
    return df[df["t_min"] >= t_end - window_min].copy()

def linear_rate_k_per_hour(df: pd.DataFrame, minutes: float = 60.0) -> float:
    part = df[df["t_min"] <= minutes]
    if len(part) < 2:
        return np.nan
    x = part["t_min"].to_numpy()
    y = part["Sample Temp Smooth"].to_numpy()
    slope_k_per_min = np.polyfit(x, y, 1)[0]
    return slope_k_per_min * 60.0

def summary_metrics(df: pd.DataFrame) -> dict:
    tail = last_window(df, TAIL_WINDOW_MIN)

    return {
        "duration_min": df["t_min"].max(),
        "sample_start_K": df["Sample Temp"].iloc[0],
        "sample_end_K": df["Sample Temp"].iloc[-1],
        "wall_start_K": df["Wall Temp"].iloc[0],
        "wall_end_K": df["Wall Temp"].iloc[-1],
        "sample_tail_mean_K": tail["Sample Temp"].mean(),
        "wall_tail_mean_K": tail["Wall Temp"].mean(),
        "gap_tail_mean_K": tail["gap"].mean(),
        "gap_tail_std_K": tail["gap"].std(),
        "rate_first_60min_K_per_h": linear_rate_k_per_hour(df, 60.0),
        "max_sample_temp_K": df["Sample Temp"].max(),
        "max_wall_temp_K": df["Wall Temp"].max(),
    }

def print_case_summary(name: str, metrics: dict):
    print(f"\n=== {name.upper()} ===")
    for k, v in metrics.items():
        if isinstance(v, (float, np.floating)):
            print(f"{k:28s}: {v: .6f}")
        else:
            print(f"{k:28s}: {v}")

def compare_relative_emissivity(uncoated: pd.DataFrame, coated: pd.DataFrame):
    """
    Relative comparison only:
    lower steady-state sample temperature and smaller sample-wall gap indicate higher emissivity.
    """
    u_tail = last_window(uncoated, TAIL_WINDOW_MIN)
    c_tail = last_window(coated, TAIL_WINDOW_MIN)

    u_gap = u_tail["gap"].mean()
    c_gap = c_tail["gap"].mean()

    u_Ts = u_tail["Sample Temp"].mean()
    c_Ts = c_tail["Sample Temp"].mean()

    # A simple relative score:
    # smaller gap -> higher emissivity, so use the negative gap as a proxy
    u_score = -u_gap
    c_score = -c_gap

    rel_change = (c_score - u_score) / (abs(u_score) + 1e-12) * 100.0

    return {
        "uncoated_gap_mean": u_gap,
        "coated_gap_mean": c_gap,
        "uncoated_sample_mean": u_Ts,
        "coated_sample_mean": c_Ts,
        "uncoated_emissivity_proxy": u_score,
        "coated_emissivity_proxy": c_score,
        "relative_proxy_change_percent": rel_change,
        "higher_emissivity_case": "coated" if c_score > u_score else "uncoated",
    }

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
uncoated = load_case(UNCOATED_FILE, "uncoated")
coated = load_case(COATED_FILE, "coated")

# ------------------------------------------------------------
# Print summaries
# ------------------------------------------------------------
u_metrics = summary_metrics(uncoated)
c_metrics = summary_metrics(coated)

print_case_summary("uncoated", u_metrics)
print_case_summary("coated", c_metrics)

comparison = compare_relative_emissivity(uncoated, coated)

print("\n=== RELATIVE COMPARISON ===")
print(f"Uncoated mean gap in last {TAIL_WINDOW_MIN} min : {comparison['uncoated_gap_mean']:.4f} K")
print(f"Coated mean gap in last {TAIL_WINDOW_MIN} min   : {comparison['coated_gap_mean']:.4f} K")
print(f"Uncoated mean sample temp (tail)               : {comparison['uncoated_sample_mean']:.4f} K")
print(f"Coated mean sample temp (tail)                 : {comparison['coated_sample_mean']:.4f} K")
print(f"Relative emissivity proxy change               : {comparison['relative_proxy_change_percent']:.2f} %")
print(f"Conclusion                                     : {comparison['higher_emissivity_case']} sample is more emissive")

# ------------------------------------------------------------
# Plot 1: sample and wall temperatures
# ------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(uncoated["t_min"], uncoated["Sample Temp Smooth"], label="Uncoated sample")
plt.plot(uncoated["t_min"], uncoated["Wall Temp Smooth"], "--", label="Uncoated wall")
plt.plot(coated["t_min"], coated["Sample Temp Smooth"], label="Coated sample")
plt.plot(coated["t_min"], coated["Wall Temp Smooth"], "--", label="Coated wall")
plt.xlabel("Time [min]")
plt.ylabel("Temperature [K]")
plt.title("Temperature history")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Plot 2: temperature difference sample - wall
# ------------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(uncoated["t_min"], uncoated["gap_smooth"], label="Uncoated: sample - wall")
plt.plot(coated["t_min"], coated["gap_smooth"], label="Coated: sample - wall")
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Time [min]")
plt.ylabel("ΔT [K]")
plt.title("Thermal lag relative to wall")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Plot 3: steady-state tail comparison
# ------------------------------------------------------------
u_tail = last_window(uncoated, TAIL_WINDOW_MIN)
c_tail = last_window(coated, TAIL_WINDOW_MIN)

plt.figure(figsize=(12, 4))
plt.plot(u_tail["t_min"], u_tail["Sample Temp Smooth"], label="Uncoated sample (tail)")
plt.plot(c_tail["t_min"], c_tail["Sample Temp Smooth"], label="Coated sample (tail)")
plt.plot(u_tail["t_min"], u_tail["Wall Temp Smooth"], "--", label="Uncoated wall (tail)")
plt.plot(c_tail["t_min"], c_tail["Wall Temp Smooth"], "--", label="Coated wall (tail)")
plt.xlabel("Time [min]")
plt.ylabel("Temperature [K]")
plt.title(f"Final {TAIL_WINDOW_MIN} min comparison")
plt.legend()
plt.tight_layout()
plt.show()
