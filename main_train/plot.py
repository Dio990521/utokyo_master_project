import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import ast

VERSION = "20251130_pen1x1_width1_threshold14_attention_test"
PLOT_VALIDATION_DATA = False
PLOT_PAINTED_PIXELS_TOGETHER = False
PLOT_MAX_STROKE_LENGTH = True
PLOT_AVG_STROKE_LENGTH = False
PLOT_COMBO_METRICS = True
COLUMN_TO_PLOT = "similarity"  # similarity, used_budgets, block_similarity, block_reward, step_rewards
TRAIN_WINDOW_SIZE = 100



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
        # ---------------------------

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
        # ---------------------------

        plt.figure(figsize=(15, 7))

        # plt.plot(df['episode'], df['avg_combo_len'], alpha=0.4, color='gray', label='Raw Avg Stroke Length')
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

        if COLUMN_TO_PLOT == "similarity":
            metrics_to_plot = ["precision", "recall_black", "recall_white"]
            plot_styles = {
                "pixel_similarity": 'r-',
                "similarity": 'b-',
                "recall_black": 'g--',
                "recall_white": 'm:',
                "balanced_accuracy": 'c-.',
            }
            default_style = 'k-'
            plot_labels = {
                "pixel_similarity": f'Pixel Accuracy (MA {TRAIN_WINDOW_SIZE})',
                "precision": f'Precision (MA {TRAIN_WINDOW_SIZE})',
                "recall_black": f'Black Recall (MA {TRAIN_WINDOW_SIZE})',
                "recall_white": f'White Recall (MA {TRAIN_WINDOW_SIZE})'
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
                plt.plot(df.index, metric_ma, style, label=plot_labels.get(col, col))

            plt.title(f"Training Performance: Similarity Metrics (Moving Average) - {VERSION}")
            plt.xlabel("Episode")
            plt.ylabel("Metric Value (0-1)")
            plt.ylim(-0.05, 1.05)

        elif PLOT_PAINTED_PIXELS_TOGETHER:
            required_columns = ["total_painted", "correctly_painted"]
            if not all(col in df.columns for col in required_columns):
                print(f"Error: Missing required columns in data: {required_columns}")
                plt.close()
                return

            precision = np.divide(
                df["correctly_painted"],
                df["total_painted"],
                out=np.zeros_like(df["correctly_painted"], dtype=float),
                where=df["total_painted"] != 0
            )
            precision = np.nan_to_num(precision, nan=0.0, posinf=0.0, neginf=0.0)

            # Calculate moving average of precision
            precision_ma = pd.Series(precision).rolling(window=TRAIN_WINDOW_SIZE).mean()
            recall_black_ma = df["recall_black"].rolling(window=TRAIN_WINDOW_SIZE).mean()

            print(f"Final Painting Precision (MA {TRAIN_WINDOW_SIZE}): {precision_ma.iloc[-1]:.4f}")
            print(f"Final Black Recall (MA {TRAIN_WINDOW_SIZE}): {recall_black_ma.iloc[-1]:.4f}")

            # Plot both metrics
            plt.plot(df.index, precision_ma, 'r-',
                     label=f'Painting Precision (MA {TRAIN_WINDOW_SIZE})')  # Red solid line
            plt.plot(df.index, recall_black_ma, 'g--', label=f'Black Recall (MA {TRAIN_WINDOW_SIZE})')

            plt.title(f"Training Performance: Precision vs Black Recall (Moving Average) - {VERSION}")
            plt.xlabel("Episode")
            plt.ylabel("Metric Value (0-1)")
            plt.ylim(-0.05, 1.05)

        else:
            if COLUMN_TO_PLOT not in df.columns:
                print(f"Error: Column '{COLUMN_TO_PLOT}' not found in training data.")
                plt.close()
                return

            data_to_plot = df[COLUMN_TO_PLOT].rolling(window=TRAIN_WINDOW_SIZE).mean()

            last_val = data_to_plot.iloc[-1]
            label_text = COLUMN_TO_PLOT.replace('_', ' ').title()
            print(f"Final {label_text} (MA {TRAIN_WINDOW_SIZE}): {last_val:.4f}")

            plt.plot(df.index, data_to_plot, label=f'{label_text} (MA {TRAIN_WINDOW_SIZE})')

            plt.title(f"Training Performance: {label_text} (Moving Average) - {VERSION}")
            plt.xlabel("Episode")
            plt.ylabel(label_text)

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    except Exception as e:
        print(f"Error plotting training data: {e}")
        if 'plt' in locals() and plt.gcf().get_axes():
            plt.close()


def plot_validation_data():
    DATA_PATH = os.path.join(f"../training_outputs/{VERSION}/", "validation_data.csv")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Validation data file not found at {DATA_PATH}")
        return

    try:
        df = pd.read_csv(DATA_PATH)
        plt.figure(figsize=(15, 7))

        if COLUMN_TO_PLOT == "similarity":
            metrics_to_plot = ["precision", "recall_black", "recall_white"]
            plot_styles = {
                "pixel_similarity": 'r-o',  # Added markers 'o' for validation points
                "precision": 'b-o',
                "recall_black": 'g--o',
                "recall_white": 'm:o',
                "balanced_accuracy": 'c-.o',
            }
            default_style = 'k-o'
            plot_labels = {
                "pixel_similarity": 'Mean Pixel Accuracy',
                "precision": 'Mean precision',  # Changed key
                "recall_black": 'Mean Black Recall',
                "recall_white": 'Mean White Recall',
                "balanced_accuracy": 'Mean Balanced Accuracy'
            }

            # Check which columns are available in validation data
            available_metrics = [col for col in metrics_to_plot if col in df.columns]

            if not available_metrics:
                print(f"Error: None of the specified similarity metrics ({metrics_to_plot}) found in {DATA_PATH}.")
                plt.close()
                return

            print(f"Plotting available validation similarity metrics: {available_metrics}")

            # Group by 'step' and calculate mean for each available metric
            grouped_data = df.groupby('step')

            for col in available_metrics:
                metric_mean = grouped_data[col].mean()

                last_val = metric_mean.iloc[-1]
                print(f"Final Validation Mean {col}: {last_val:.4f}")

                style = plot_styles.get(col, default_style)
                # Plot mean value against the step index
                plt.plot(metric_mean.index, metric_mean.values, style, label=plot_labels.get(col, f'Mean {col}'),
                         markersize=6)

            plt.title(f"Validation Performance: Mean Similarity Metrics - {VERSION}")
            plt.xlabel("Training Timestep")
            plt.ylabel("Mean Metric Value (0-1)")
            plt.ylim(-0.05, 1.05)
        elif PLOT_PAINTED_PIXELS_TOGETHER:

            grouped_data = df.groupby('step')
            total_painted_ma = grouped_data["total_painted"].mean()
            correctly_painted_ma = grouped_data["correctly_painted"].mean()

            print(f"Final Validation Total Painted: {total_painted_ma.iloc[-1]:.4f}")
            print(f"Final Validation Correctly Painted: {correctly_painted_ma.iloc[-1]:.4f}")

            plt.plot(total_painted_ma.index, total_painted_ma.values,
                     label=f'Total Painted Pixels (MA {TRAIN_WINDOW_SIZE})')
            plt.plot(correctly_painted_ma.index, correctly_painted_ma.values,
                     label=f'Correctly Painted Pixels (MA {TRAIN_WINDOW_SIZE})')

            plt.title(f"Validation Performance: Painted Pixels Count - {VERSION}")
            plt.xlabel("Training Timestep")
            plt.ylabel("Number of Pixels")
        else:
            COLUMN_TO_PLOT_VAL = COLUMN_TO_PLOT  # Use the same column as specified
            if COLUMN_TO_PLOT_VAL not in df.columns:
                print(f"Error: Column '{COLUMN_TO_PLOT_VAL}' not found in validation data.")
                plt.close()
                return

            val_agg = df.groupby('step')[COLUMN_TO_PLOT_VAL].mean()

            last_val = val_agg.iloc[-1]
            label_text = COLUMN_TO_PLOT_VAL.replace('_', ' ').title()
            print(f"Final Validation Mean {label_text}: {last_val:.4f}")

            plt.plot(val_agg.index, val_agg.values, 'o-', label=f'Mean {label_text}',
                     markersize=6)  # Default style for single metric

            plt.title(f"Validation Performance: Mean {label_text} - {VERSION}")
            plt.xlabel("Training Timestep")
            plt.ylabel(f"Mean {label_text}")

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    except Exception as e:
        print(f"Error plotting validation data: {e}")
        if 'plt' in locals() and plt.gcf().get_axes():
            plt.close()


if __name__ == '__main__':
    if PLOT_MAX_STROKE_LENGTH:
        print(f"Plotting MAX STROKE LENGTH for version: {VERSION}")
        plot_max_stroke_length(os.path.join(f"../training_outputs/{VERSION}/", "training_data.csv"),
                               window_size=TRAIN_WINDOW_SIZE)
    elif PLOT_AVG_STROKE_LENGTH:
        plot_average_combo(os.path.join(f"../training_outputs/{VERSION}/", "training_data.csv"),
                           window_size=TRAIN_WINDOW_SIZE)
    elif PLOT_VALIDATION_DATA:
        print(f"Plotting VALIDATION data for version: {VERSION}")
        plot_validation_data()
    else:
        # Update the message based on the plotting mode
        if COLUMN_TO_PLOT == "similarity":
            print(f"Plotting TRAINING data (Multiple Similarity Metrics) for version: {VERSION}")
        else:
            print(f"Plotting TRAINING data (Metric: {COLUMN_TO_PLOT}) for version: {VERSION}")
        plot_training_data()