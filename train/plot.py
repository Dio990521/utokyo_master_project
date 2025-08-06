import pandas as pd
import matplotlib.pyplot as plt

OPTION = "similarity" # "used_budget"
df = pd.read_csv("similarity_test_norm_obs_1.csv", header=None)
df.columns = ['similarity']
df['step'] = df.index

window_size = 100
df['moving_avg'] = df['similarity'].rolling(window=window_size).mean()

plt.figure(figsize=(10, 5))
if OPTION == "used_budget":
    plt.plot(df['step'], df['similarity'], color='blue')
else:
    plt.plot(df['step'], df['moving_avg'], label=f'{window_size}-step Moving Average', color='blue')

plt.xlabel("Episode")
if OPTION == "similarity":
    plt.ylabel("Similarity")
    plt.title("Similarity between the target sketch and the final canvas status")
else:
    plt.ylabel("Used strokes")
    plt.title("Used strokes for each episode")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()