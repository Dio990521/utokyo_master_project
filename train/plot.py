import pandas as pd
import matplotlib.pyplot as plt
import os

VERSION = "_test"

# ('similarity', 'used_budgets')
COLUMN_TO_PLOT = "similarity"

DATA_PATH = f"../training_outputs/{VERSION}/episode_data.csv"

if not os.path.exists(DATA_PATH):
    print(f"Error: Cannot find the data file at {DATA_PATH}")
    exit()

df = pd.read_csv(DATA_PATH, index_col="episode")

if COLUMN_TO_PLOT not in df.columns:
    print(f"Error: Column '{COLUMN_TO_PLOT}' not found in the data file.")
    print(f"Available columns are: {list(df.columns)}")
    exit()

window_size = 100
df['moving_avg'] = df[COLUMN_TO_PLOT].rolling(window=window_size).mean()
plt.figure(figsize=(12, 6))

plt.plot(df.index, df['moving_avg'], label=f'{window_size}-Episode Moving Average', color='blue')
# plt.plot(df.index, df[COLUMN_TO_PLOT], color='gray', alpha=0.3, label='Raw Data')

plt.xlabel("Episode")
plt.ylabel(COLUMN_TO_PLOT.replace('_', ' ').title())
plt.title(f"{COLUMN_TO_PLOT.replace('_', ' ').title()} during Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()