import pandas as pd
import matplotlib.pyplot as plt
import os

VERSION = "20251023_2squares_1_redo"
PLOT_VALIDATION_DATA = False
PLOT_PAINTED_PIXELS_TOGETHER = True
COLUMN_TO_PLOT = "similarity"  # similarity, used_budgets, block_similarity, block_reward, step_rewards
TRAIN_WINDOW_SIZE = 100

def plot_training_data():
    DATA_PATH = os.path.join(f"../training_outputs/{VERSION}/", "training_data.csv")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Training data file not found at {DATA_PATH}")
        return

    try:
        df = pd.read_csv(DATA_PATH)
        plt.figure(figsize=(15, 7))

        if PLOT_PAINTED_PIXELS_TOGETHER:
            required_columns = ["total_painted", "correctly_painted"]
            if not all(col in df.columns for col in required_columns):
                print(f"Error: Missing required columns in data: {required_columns}")
                plt.close()
                return

            total_painted_ma = df["total_painted"].rolling(window=TRAIN_WINDOW_SIZE).mean()
            correctly_painted_ma = df["correctly_painted"].rolling(window=TRAIN_WINDOW_SIZE).mean()

            plt.plot(df.index, total_painted_ma, label=f'Total Painted Pixels (MA {TRAIN_WINDOW_SIZE})')
            plt.plot(df.index, correctly_painted_ma, label=f'Correctly Painted Pixels (MA {TRAIN_WINDOW_SIZE})')

            plt.title(f"Training Performance: Painted Pixels Count (Moving Average) - {VERSION}")
            plt.xlabel("Episode")
            plt.ylabel("Number of Pixels")
            plot_filename = "painted_pixels_performance.png"

        else:
            # --- Plot a single metric specified by COLUMN_TO_PLOT ---
            if COLUMN_TO_PLOT not in df.columns:
                print(f"Error: Column '{COLUMN_TO_PLOT}' not found in training data.")
                plt.close() # Close the plot window
                return

            data_to_plot = df[COLUMN_TO_PLOT].rolling(window=TRAIN_WINDOW_SIZE).mean()
            label_text = COLUMN_TO_PLOT.replace('_', ' ').title()
            plt.plot(df.index, data_to_plot, label=f'{label_text} (MA {TRAIN_WINDOW_SIZE})')

            plt.title(f"Training Performance: {label_text} (Moving Average) - {VERSION}")
            plt.xlabel("Episode")
            plt.ylabel(label_text)

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show() # Uncomment to display the plot locally
        plt.close() # Close the plot figure to free memory

    except Exception as e:
        print(f"Error plotting training data: {e}")
        if 'plt' in locals() and plt.gcf().get_axes(): # Check if a figure exists
             plt.close()


def plot_validation_data():
    DATA_PATH = os.path.join(f"../training_outputs/{VERSION}/", "validation_data.csv")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Validation data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if COLUMN_TO_PLOT not in df.columns:
        print(f"Error: Column '{COLUMN_TO_PLOT}' not found in validation data.")
        return

    plt.figure(figsize=(15, 7))

    val_agg = df.groupby('step')[COLUMN_TO_PLOT].mean()

    plt.plot(val_agg.index, val_agg.values, 'o-', label='Validation Score', markersize=6)
    plt.scatter(val_agg.index, val_agg.values)

    plt.title(f"Validation Performance: {COLUMN_TO_PLOT.replace('_', ' ').title()}")
    plt.xlabel("Training Timestep")
    plt.ylabel(COLUMN_TO_PLOT.replace('_', ' ').title())
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if PLOT_VALIDATION_DATA:
        print(f"Plotting VALIDATION data for version: {VERSION}")
        plot_validation_data()
    else:
        print(f"Plotting TRAINING data for version: {VERSION}")
        plot_training_data()