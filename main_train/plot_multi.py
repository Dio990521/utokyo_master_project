import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
# List of experiments to compare: (folder_name, legend_label)
EXPERIMENTS = [
    ("final5_obs_r_action_j_jump_nopt", "no pt"),
    ("final5_obs_r_action_j_09pt", "jump -0.25pt"),
    ("final5_obs_r_action_j_jump_05pt", "jump -0.5pt"),
    ("final5_obs_r_action_j_jump_075pt", "jump -0.75pt"),
]

# Metric Selection
# Options include: precision, recall_black, f1_score, jump_ratio
COLUMN_TO_PLOT = "precision"
NAME = "Precision of Drawn Pixels"

# Plotting Parameters
TRAIN_WINDOW_SIZE = 100
BASE_OUTPUT_DIR = "../training_outputs/"
ENABLE_TRUNCATION = False  # If True, cuts all plots to the length of the shortest run

DISTINCT_COLORS = [
    '#000000',  # Black
    '#D55E00',  # Vermilion
    '#0072B2',  # Blue
    '#999999',  # Grey
    '#E69F00',  # Orange
    '#009E73',  # Bluish Green
    '#CC79A7',  # Reddish Purple
    '#56B4E9',  # Sky Blue
    '#F0E442'  # Yellow
]


# ==========================================
# 2. PLOTTING FUNCTION
# ==========================================
def plot_multi_comparison():
    plt.figure(figsize=(15, 8))
    print(f"--- Starting Comparison Plot for {COLUMN_TO_PLOT} ---")

    loaded_datasets = []
    min_episode_count = float('inf')
    min_max_steps = float('inf')
    use_step_axis = True

    # --- Data Loading Phase ---
    for i, (version, label) in enumerate(EXPERIMENTS):
        data_path = os.path.join(BASE_OUTPUT_DIR, version, "training_data.csv")

        if not os.path.exists(data_path):
            print(f"[Warning] File not found for version '{version}': {data_path}")
            continue

        try:
            df = pd.read_csv(data_path)

            # Validate required columns
            if COLUMN_TO_PLOT == "jump_ratio":
                if "jump_draw_combo_count" not in df.columns or "target_pixel_count" not in df.columns:
                    print(f"[Warning] Necessary columns for jump_ratio not found in experiment: {version}")
                    continue
            elif COLUMN_TO_PLOT not in df.columns:
                print(f"[Warning] Column '{COLUMN_TO_PLOT}' not found in experiment: {version}")
                continue

            # Determine X-axis type (Steps vs Episodes) and track minimum lengths
            if "total_steps" not in df.columns:
                use_step_axis = False
            else:
                current_max_step = df["total_steps"].max()
                if current_max_step < min_max_steps:
                    min_max_steps = current_max_step

            if len(df) < min_episode_count:
                min_episode_count = len(df)

            loaded_datasets.append({
                "df": df,
                "label": label,
                "color": DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
            })

        except Exception as e:
            print(f"[Error] Failed to process {version}: {e}")

    # --- Plotting Phase ---
    if loaded_datasets:
        # Define X-axis label
        if use_step_axis:
            print(f"--- Aligning all plots to shortest duration: {min_max_steps} Steps ---")
            x_label = "Total Training Steps"
        else:
            print(f"--- Aligning all plots to shortest duration: {min_episode_count} Episodes ---")
            x_label = "Episode"

        for item in loaded_datasets:
            df = item["df"]
            label = item["label"]
            color = item["color"]

            # Calculate the metric data
            if COLUMN_TO_PLOT == "jump_ratio":
                safe_pixel_count = df['target_pixel_count'].replace(0, 1)
                data_series = df['jump_draw_combo_count'] / safe_pixel_count
            else:
                data_series = df[COLUMN_TO_PLOT]

            # Apply rolling average
            data_to_plot = data_series.rolling(window=TRAIN_WINDOW_SIZE).mean()

            # Set X-axis data
            if use_step_axis:
                x_axis = df["total_steps"]
            else:
                x_axis = df.index

            # Apply Truncation if enabled
            if ENABLE_TRUNCATION:
                if use_step_axis:
                    mask = x_axis <= min_max_steps
                    x_axis = x_axis[mask]
                    data_to_plot = data_to_plot[mask]
                else:
                    x_axis = x_axis[:min_episode_count]
                    data_to_plot = data_to_plot.iloc[:min_episode_count]

            # Print final value for logging
            last_val = data_to_plot.dropna().iloc[-1] if not data_to_plot.dropna().empty else 0.0
            print(f"[{label}] Final {COLUMN_TO_PLOT} (MA {TRAIN_WINDOW_SIZE}): {last_val:.4f}")

            # Plot the line
            plt.plot(
                x_axis,
                data_to_plot,
                label=f"{label} (Final: {last_val:.2f})",
                linewidth=2.5,
                color=color,
                alpha=0.9
            )

        # --- Final Plot Styling ---
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(NAME, fontsize=12)

        # Set fixed limits for ratio/probability metrics
        if COLUMN_TO_PLOT in ["pixel_similarity", "recall_black", "recall_grey", "recall_white", "precision",
                              "f1_score", "jump_ratio"]:
            plt.ylim(-0.05, 1.05)

        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend(fontsize=12, loc='best')
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo valid data found to plot.")


# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    plot_multi_comparison()