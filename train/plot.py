import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("similarity_2.csv", header=None)
df.columns = ['similarity']
df['step'] = df.index

window_size = 100
df['moving_avg'] = df['similarity'].rolling(window=window_size).mean()

plt.figure(figsize=(10, 5))
#plt.plot(df['step'], df['similarity'], label='Raw Similarity', color='gray', alpha=0.3)
plt.plot(df['step'], df['moving_avg'], label=f'{window_size}-step Moving Average', color='blue')

plt.xlabel("Episode")
plt.ylabel("Similarity")
plt.title("Similarity between the target sketch and the final canvas status")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()