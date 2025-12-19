import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

EXPERIMENTS = [
    ("final_action2_obs2", "A2 + O2 enable jump penalty"),
    ("final_action2_obs2_no_jump_penalty", "A2 + O2 disable jump penalty"),
    #("final_action1_obs1_no_threshold", "A1 + O1"),
    #("final_action1_obs2_no_threshold", "A1 + O2"),
    #("final_action2_obs1_no_threshold", "A2 + O1"),
    #("final_action2_obs2_no_threshold", "A2 + O2"),
    #("final_action2_obs1_no_jump_penalty", "A2 + O1"),
    #("final_action2_obs2_no_jump_penalty", "A2 + O2"),
    #("final_action1_obs3", "A1 + O3"),
    #("final_action2_obs3", "A2 + O3"),
]

# precision, recall_black, recall_grey, pixel_similarity, f1_score, jump_draw_combo_count
COLUMN_TO_PLOT = "precision"
#NAME = "Count of [Jump + Draw In-Place] Action Combination (Disable Penalty Threshold)"
#NAME = "Count of [Jump + Draw In-Place] Action Combination (Disable Jump Penalty)"
#NAME = "Count of [Jump + Draw In-Place] Action Combination"
#NAME = "Precision of Drawn Black Pixels (Disable Penalty Threshold)"
#NAME = "Precision of Drawn Black Pixels (Disable Jump Penalty)"
NAME = "Precision of Drawn Black Pixels"

TRAIN_WINDOW_SIZE = 100
BASE_OUTPUT_DIR = "../training_outputs/"
ENABLE_TRUNCATION = True
TITLE_NAME = f"Comparison: {NAME} (Moving Average Window = {TRAIN_WINDOW_SIZE})"

DISTINCT_COLORS = [
    '#000000',  # Black (黑色 - 用于强调)
    '#D55E00',  # Vermilion (朱红 - 非常醒目)
    '#0072B2',  # Blue (深蓝 - 经典对比)
    '#999999',  # Grey (灰色)
    '#E69F00',  # Orange (橙色 - 亮眼)
    '#009E73',  # Bluish Green (蓝绿 - 与红/蓝区分极大)
    '#CC79A7',  # Reddish Purple (紫红)
    '#56B4E9',  # Sky Blue (天蓝)
    '#F0E442'  # Yellow (黄色 - 注意：在白底上可能较浅)
]


def plot_multi_comparison():
    plt.figure(figsize=(15, 8))

    print(f"--- Starting Comparison Plot for {COLUMN_TO_PLOT} ---")

    loaded_datasets = []
    min_length = float('inf')

    for i, (version, label) in enumerate(EXPERIMENTS):
        data_path = os.path.join(BASE_OUTPUT_DIR, version, "training_data.csv")

        if not os.path.exists(data_path):
            print(f"[Warning] File not found for version '{version}': {data_path}")
            continue

        try:
            df = pd.read_csv(data_path)

            if COLUMN_TO_PLOT not in df.columns:
                print(f"[Warning] Column '{COLUMN_TO_PLOT}' not found in experiment: {version}")
                continue

            if len(df) < min_length:
                min_length = len(df)

            loaded_datasets.append({
                "df": df,
                "label": label,
                "color": DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
            })

        except Exception as e:
            print(f"[Error] Failed to process {version}: {e}")

    if loaded_datasets:
        print(f"--- Aligning all plots to shortest duration: {min_length} episodes ---")

        for item in loaded_datasets:
            df = item["df"]
            label = item["label"]
            color = item["color"]

            if ENABLE_TRUNCATION:
                df_active = df.iloc[:min_length].copy()
            else:
                df_active = df.copy()

            data_to_plot = df_active[COLUMN_TO_PLOT].rolling(window=TRAIN_WINDOW_SIZE).mean()

            last_val = data_to_plot.dropna().iloc[-1] if not data_to_plot.dropna().empty else 0.0
            print(f"[{label}] Final {COLUMN_TO_PLOT} (MA {TRAIN_WINDOW_SIZE}): {last_val:.4f}")

            plt.plot(
                df_active.index,
                data_to_plot,
                label=f"{label} (Final: {last_val:.2f})",
                linewidth=2.5,
                color=color,
                alpha=0.9
            )

        clean_name = COLUMN_TO_PLOT.replace('_', ' ').title()
        plt.title(TITLE_NAME, fontsize=16)
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel(NAME, fontsize=12)

        if COLUMN_TO_PLOT in ["pixel_similarity", "recall_black", "recall_grey", "recall_white", "precision",
                              "f1_score"]:
            plt.ylim(-0.05, 1.05)

        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend(fontsize=12, loc='best')
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo valid data found to plot.")


if __name__ == "__main__":
    plot_multi_comparison()