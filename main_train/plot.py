import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
import traceback

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
VERSION = "final5_obs_r_action_j_09pt"

# Plotting Toggles
PLOT_PAINTED_PIXELS_TOGETHER = False

# Column Selection
# Options: "similarity", "jump_ratio",
# "episode_return", "jump_count", "target_pixel_count"
COLUMN_TO_PLOT = "jump_ratio"

# Parameters
TRAIN_WINDOW_SIZE = 100
DATA_DIR = f"../training_outputs/{VERSION}/"
OUTPUT_DIR = "../figures/final5/"  # Hardcoded output path from original logic

# Plot Styling
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def safe_eval_list(val):
    """Safely parses a string representation of a list into a Python list."""
    try:
        if pd.isna(val): return []
        return ast.literal_eval(str(val))
    except (ValueError, SyntaxError):
        return []


def get_data_path():
    """Returns the full path to the training data CSV."""
    return os.path.join(DATA_DIR, "training_data.csv")


# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================
def plot_training_data():
    """Generic plotter for various training metrics (Jump Ratio, Similarity, etc.)."""
    data_path = get_data_path()
    if not os.path.exists(data_path):
        print(f"Error: Training data file not found at {data_path}")
        return

    try:
        df = pd.read_csv(data_path)
        plt.figure(figsize=(15, 7))

        # Determine X-axis
        if "total_steps" in df.columns:
            x_axis = df["total_steps"]
            xlabel_text = "Total Training Steps"
        else:
            x_axis = df.index
            xlabel_text = "Episode"

        # --- Case 1: Jump Ratio ---
        if COLUMN_TO_PLOT == "jump_ratio":
            required_cols = ['jump_draw_combo_count', 'target_pixel_count']
            if not all(col in df.columns for col in required_cols):
                print(f"Error: Missing columns {required_cols}. Cannot plot ratio.")
                plt.close()
                return

            safe_pixel_count = df['target_pixel_count'].replace(0, 1)
            ratio_series = df['jump_draw_combo_count'] / safe_pixel_count
            data_to_plot = ratio_series.rolling(window=TRAIN_WINDOW_SIZE).mean()

            print(f"Final Jump/Pixel Ratio (MA {TRAIN_WINDOW_SIZE}): {data_to_plot.iloc[-1]:.4f}")

            plt.plot(x_axis, data_to_plot, color='green', linewidth=2)
            plt.xlabel(xlabel_text)
            plt.ylabel("Jump Dependency Ratio")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- Case 2: Similarity Metrics (Precision/Recall/F1) ---
        elif COLUMN_TO_PLOT == "similarity":
            metrics_to_plot = ["precision", "recall_black", "f1_score"]
            plot_styles = {
                "f1_score": 'r:', "similarity": 'b-', "recall_black": 'b--',
                "recall_white": 'm:', "recall_all": 'c-.'
            }
            default_style = 'k-'
            plot_labels = {
                "f1_score": 'F1-Score',
                "precision": 'Precision',
                "recall_black": 'Recall',
                "recall_grey": f'Grey Recall (MA {TRAIN_WINDOW_SIZE})',
                "recall_white": f'White Recall (MA {TRAIN_WINDOW_SIZE})',
                "recall_all": f'Black+Grey Recall (MA {TRAIN_WINDOW_SIZE})'
            }

            available_metrics = [col for col in metrics_to_plot if col in df.columns]
            if not available_metrics:
                print(f"Error: Metrics {metrics_to_plot} not found in CSV.")
                plt.close()
                return

            print(f"Plotting available similarity metrics: {available_metrics}")
            for col in available_metrics:
                metric_ma = df[col].rolling(window=TRAIN_WINDOW_SIZE).mean()
                print(f"Final {col} (MA {TRAIN_WINDOW_SIZE}): {metric_ma.iloc[-1]:.4f}")

                style = plot_styles.get(col, default_style)
                plt.plot(x_axis, metric_ma, style, label=plot_labels.get(col, col))

            plt.xlabel(xlabel_text)
            plt.ylabel("Metric Value")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.ylim(-0.05, 1.05)
            plt.legend()

            # --- Case 3: Generic Column Plot ---
        else:
            if COLUMN_TO_PLOT not in df.columns:
                print(f"Error: Column '{COLUMN_TO_PLOT}' not found in training data.")
                plt.close()
                return

            data_to_plot = df[COLUMN_TO_PLOT].rolling(window=TRAIN_WINDOW_SIZE).mean()
            label_text = COLUMN_TO_PLOT.replace('_', ' ').title()

            print(f"Final {label_text} (MA {TRAIN_WINDOW_SIZE}): {data_to_plot.iloc[-1]:.4f}")
            plt.plot(x_axis, data_to_plot, label=f'{label_text} (MA {TRAIN_WINDOW_SIZE})')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.xlabel(xlabel_text)
            plt.ylabel(label_text)

        plt.tight_layout()

        # Create output directory if it doesn't exist (Optional safety)
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)

        plt.savefig(os.path.join(OUTPUT_DIR, "figure.pdf"), bbox_inches="tight")
        plt.show()
        plt.close()

    except Exception as e:
        print(f"Error plotting training data: {e}")
        traceback.print_exc()
        if 'plt' in locals() and plt.gcf().get_axes():
            plt.close()


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    data_path = get_data_path()

    # Default fallback: Plot generic training metrics
    if COLUMN_TO_PLOT == "similarity":
        print(f"Plotting TRAINING data (Multiple Similarity Metrics) for version: {VERSION}")
    elif COLUMN_TO_PLOT == "jump_ratio":
        print(f"Plotting TRAINING data (Jump/Pixel Ratio) for version: {VERSION}")
    else:
        print(f"Plotting TRAINING data (Metric: {COLUMN_TO_PLOT}) for version: {VERSION}")

    plot_training_data()