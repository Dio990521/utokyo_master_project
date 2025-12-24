import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import ast

VERSION = "final4_obs_r_action_j"
PLOT_VALIDATION_DATA = False
PLOT_PAINTED_PIXELS_TOGETHER = False
PLOT_MAX_STROKE_LENGTH = False
PLOT_AVG_STROKE_LENGTH = False
PLOT_COMBO_METRICS = False

# Options: "similarity", "jump_ratio", "jump_draw_combo_count", "negative_reward", "jump_count", "target_pixel_count"
COLUMN_TO_PLOT = "jump_ratio"
# ===============================

TRAIN_WINDOW_SIZE = 100


def safe_eval_list(val):
    try:
        if pd.isna(val): return []
        return ast.literal_eval(str(val))
    except (ValueError, SyntaxError):
        return []


def plot_combo_metrics(data_path, window_size=100):
    if not os.path.exists(data_path):
        print(f"Error: Training data file not found at {data_path}")
        return

    try:
        df = pd.read_csv(data_path)
        print(f"Loaded data with {len(df)} episodes.")

        if 'episode_base_reward' in df.columns and 'episode_combo_bonus' in df.columns:
            plt.figure(figsize=(15, 7))

            base_ma = df['episode_base_reward'].rolling(window=window_size).mean()
            bonus_ma = df['episode_combo_bonus'].rolling(window=window_size).mean()

            plt.plot(df.index, base_ma, color='#1f77b4', linewidth=2, label=f'Base Reward (No Bonus) MA{window_size}')
            plt.plot(df.index, bonus_ma, color='#ff7f0e', linewidth=2, linestyle='--',
                     label=f'Combo Bonus (Added Part) MA{window_size}')

            print(f"Final Base Reward (MA): {base_ma.iloc[-1]:.4f}")
            print(f"Final Combo Bonus (MA): {bonus_ma.iloc[-1]:.4f}")

            plt.title(f"Reward Analysis: Base Reward vs Combo Bonus - {VERSION}", fontsize=14)
            plt.xlabel("Episode")
            plt.ylabel("Reward Value")
            plt.legend(loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
        else:
            print("Skipping Reward Plot: Missing 'episode_base_reward' or 'episode_combo_bonus' columns.")

        if 'episode_combo_log' in df.columns and 'combo_sustained' in df.columns:
            df['strict_list'] = df['episode_combo_log'].apply(safe_eval_list)
            df['sustained_list'] = df['combo_sustained'].apply(safe_eval_list)

            plt.figure(figsize=(15, 7))

            df['avg_strict_len'] = df['strict_list'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
            df['avg_sustained_len'] = df['sustained_list'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)

            strict_avg_ma = df['avg_strict_len'].rolling(window=window_size).mean()
            sustained_avg_ma = df['avg_sustained_len'].rolling(window=window_size).mean()

            plt.plot(df.index, sustained_avg_ma, color='#2ca02c', linewidth=2,
                     label=f'Sustained Combo (ki re nai) - Avg Len MA{window_size}')
            plt.plot(df.index, strict_avg_ma, color='#d62728', linewidth=2, linestyle='--',
                     label=f'Strict Combo (ki re ru) - Avg Len MA{window_size}')

            print(f"Final Avg Sustained Combo Len (MA): {sustained_avg_ma.iloc[-1]:.4f}")
            print(f"Final Avg Strict Combo Len (MA): {strict_avg_ma.iloc[-1]:.4f}")

            plt.title(f"Combo Logic Comparison: Average Combo Length - {VERSION}", fontsize=14)
            plt.xlabel("Episode")
            plt.ylabel("Average Combo Length")
            plt.legend(loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
            plt.figure(figsize=(15, 7))

            df['max_strict_len'] = df['strict_list'].apply(lambda x: np.max(x) if len(x) > 0 else 0)
            df['max_sustained_len'] = df['sustained_list'].apply(lambda x: np.max(x) if len(x) > 0 else 0)

            strict_max_ma = df['max_strict_len'].rolling(window=window_size).mean()
            sustained_max_ma = df['max_sustained_len'].rolling(window=window_size).mean()

            plt.plot(df.index, sustained_max_ma, color='#9467bd', linewidth=2,
                     label=f'Sustained Combo (ki re nai) - Max Len MA{window_size}')
            plt.plot(df.index, strict_max_ma, color='#8c564b', linewidth=2, linestyle='--',
                     label=f'Strict Combo (ki re ru) - Max Len MA{window_size}')

            print(f"Final Max Sustained Combo Len (MA): {sustained_max_ma.iloc[-1]:.4f}")
            print(f"Final Max Strict Combo Len (MA): {strict_max_ma.iloc[-1]:.4f}")

            plt.title(f"Combo Logic Comparison: Max Stroke Length per Episode - {VERSION}", fontsize=14)
            plt.xlabel("Episode")
            plt.ylabel("Max Combo Length (Count)")
            plt.legend(loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

        else:
            print("Skipping Combo Plots: Missing 'episode_combo_log' or 'combo_sustained' columns.")

    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        import traceback
        traceback.print_exc()


def plot_max_stroke_length(data_path, window_size=100):
    try:
        df = pd.read_csv(data_path)

        def parse_log(log_str):
            try:
                if pd.isna(log_str): return []
                return ast.literal_eval(str(log_str))
            except:
                return []

        df['combo_list'] = df['episode_combo_log'].apply(parse_log)

        def get_max_combo(lst):
            if not lst: return 0.0
            return np.max(lst)

        df['max_stroke_len'] = df['combo_list'].apply(get_max_combo)
        df['max_stroke_ma'] = df['max_stroke_len'].rolling(window=window_size).mean()

        last_val = df['max_stroke_ma'].iloc[-1]
        print(f"Final Max Stroke Length (MA {window_size}): {last_val:.4f}")

        plt.figure(figsize=(15, 7))
        plt.plot(df['episode'], df['max_stroke_len'], alpha=0.2, color='gray', label='Raw Max Stroke Length')
        plt.plot(df['episode'], df['max_stroke_ma'], color='purple', linewidth=2,
                 label=f'Max Stroke Length (MA {window_size})')
        plt.title(f"Training Performance: Max Stroke Length per Episode - {VERSION}")
        plt.xlabel("Episode")
        plt.ylabel("Max Stroke Length (Combo Count)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting max stroke data: {e}")


def plot_average_combo(data_path, window_size=100):
    try:
        df = pd.read_csv(data_path)

        def parse_log(log_str):
            try:
                if pd.isna(log_str): return []
                return ast.literal_eval(str(log_str))
            except:
                return []

        df['combo_list'] = df['episode_combo_log'].apply(parse_log)

        def get_mean_combo(lst):
            if not lst: return 0.0
            return np.mean(lst)

        df['avg_combo_len'] = df['combo_list'].apply(get_mean_combo)
        df['avg_combo_ma'] = df['avg_combo_len'].rolling(window=window_size).mean()
        last_val = df['avg_combo_ma'].iloc[-1]
        print(f"Final Mean Stroke Length (MA {window_size}): {last_val:.4f}")
        plt.figure(figsize=(15, 7))
        plt.plot(df['episode'], df['avg_combo_ma'], color='blue', linewidth=2,
                 label=f'Mean Stroke Length (MA {window_size})')
        plt.title(f"Training Performance: Average Stroke Length per Episode - {VERSION}")
        plt.xlabel("Episode")
        plt.ylabel("Average Stroke Length")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting combo data: {e}")


def plot_training_data():
    DATA_PATH = os.path.join(f"../training_outputs/{VERSION}/", "training_data.csv")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Training data file not found at {DATA_PATH}")
        return
    try:
        df = pd.read_csv(DATA_PATH)
        plt.figure(figsize=(15, 7))

        if "total_steps" in df.columns:
            x_axis = df["total_steps"]
            xlabel_text = "Total Training Steps"
        else:
            x_axis = df.index
            xlabel_text = "Episode"

        if COLUMN_TO_PLOT == "jump_ratio":
            required_cols = ['jump_draw_combo_count', 'target_pixel_count']
            if not all(col in df.columns for col in required_cols):
                print(f"Error: Missing columns {required_cols} in data. Cannot plot ratio.")
                plt.close()
                return

            safe_pixel_count = df['target_pixel_count'].replace(0, 1)
            ratio_series = df['jump_draw_combo_count'] / safe_pixel_count

            data_to_plot = ratio_series.rolling(window=TRAIN_WINDOW_SIZE).mean()
            last_val = data_to_plot.iloc[-1]

            print(f"Final Jump/Pixel Ratio (MA {TRAIN_WINDOW_SIZE}): {last_val:.4f}")

            plt.plot(x_axis, data_to_plot, color='green',
                     linewidth=2)

            plt.title(f"Ratio of [Jump & Draw In-Place] count to the number of target black pixels")
            plt.xlabel(xlabel_text)
            plt.ylabel("Ratio")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        elif COLUMN_TO_PLOT == "similarity":
            metrics_to_plot = ["precision", "recall_black"]
            plot_styles = {"recall_grey": 'r-', "similarity": 'b-', "recall_black": 'b--', "recall_white": 'm:',
                           "recall_all": 'c-.'}
            default_style = 'k-'
            plot_labels = {
                "pixel_similarity": f'Pixel Accuracy (MA {TRAIN_WINDOW_SIZE})',
                "precision": f'Precision',
                "recall_black": f'Recall',
                "recall_grey": f'Grey Recall (MA {TRAIN_WINDOW_SIZE})',
                "recall_white": f'White Recall (MA {TRAIN_WINDOW_SIZE})',
                "recall_all": f'Black+Grey Recall (MA {TRAIN_WINDOW_SIZE})'
            }
            available_metrics = [col for col in metrics_to_plot if col in df.columns]
            if not available_metrics:
                print(f"Error: None of the specified similarity metrics ({metrics_to_plot}) found in {DATA_PATH}.")
                plt.close()
                return
            print(f"Plotting available similarity metrics: {available_metrics}")
            for col in available_metrics:
                metric_ma = df[col].rolling(window=TRAIN_WINDOW_SIZE).mean()
                last_val = metric_ma.iloc[-1]
                print(f"Final {col} (MA {TRAIN_WINDOW_SIZE}): {last_val:.4f}")
                style = plot_styles.get(col, default_style)

                plt.plot(x_axis, metric_ma, style, label=plot_labels.get(col, col))

            plt.title(
                f"Training Performance: Recall & Precision of Drawn Black Pixels")
            plt.xlabel(xlabel_text)
            plt.ylabel("Metric Value (0-1)")
            plt.ylim(-0.05, 1.05)

        elif PLOT_PAINTED_PIXELS_TOGETHER:
            pass
        else:
            if COLUMN_TO_PLOT not in df.columns:
                print(f"Error: Column '{COLUMN_TO_PLOT}' not found in training data.")
                plt.close()
                return
            data_to_plot = df[COLUMN_TO_PLOT].rolling(window=TRAIN_WINDOW_SIZE).mean()
            last_val = data_to_plot.iloc[-1]
            label_text = COLUMN_TO_PLOT.replace('_', ' ').title()
            print(f"Final {label_text} (MA {TRAIN_WINDOW_SIZE}): {last_val:.4f}")

            plt.plot(x_axis, data_to_plot, label=f'{label_text} (MA {TRAIN_WINDOW_SIZE})')

            plt.title(f"Training Performance: {label_text} (Moving Average) - {VERSION}")
            plt.xlabel(xlabel_text)
            plt.ylabel(label_text)

        if COLUMN_TO_PLOT == "similarity":
            plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Error plotting training data: {e}")
        import traceback
        traceback.print_exc()
        if 'plt' in locals() and plt.gcf().get_axes(): plt.close()


if __name__ == '__main__':
    if PLOT_COMBO_METRICS:
        plot_combo_metrics(os.path.join(f"../training_outputs/{VERSION}/", "training_data.csv"),
                           window_size=TRAIN_WINDOW_SIZE)
    elif PLOT_MAX_STROKE_LENGTH:
        print(f"Plotting MAX STROKE LENGTH for version: {VERSION}")
        plot_max_stroke_length(os.path.join(f"../training_outputs/{VERSION}/", "training_data.csv"),
                               window_size=TRAIN_WINDOW_SIZE)
    elif PLOT_AVG_STROKE_LENGTH:
        plot_average_combo(os.path.join(f"../training_outputs/{VERSION}/", "training_data.csv"),
                           window_size=TRAIN_WINDOW_SIZE)
    elif PLOT_VALIDATION_DATA:
        print(f"Plotting VALIDATION data for version: {VERSION}")
        # plot_validation_data()
    else:
        if COLUMN_TO_PLOT == "similarity":
            print(f"Plotting TRAINING data (Multiple Similarity Metrics) for version: {VERSION}")
        elif COLUMN_TO_PLOT == "jump_ratio":
            print(f"Plotting TRAINING data (Jump/Pixel Ratio) for version: {VERSION}")
        else:
            print(f"Plotting TRAINING data (Metric: {COLUMN_TO_PLOT}) for version: {VERSION}")
        plot_training_data()