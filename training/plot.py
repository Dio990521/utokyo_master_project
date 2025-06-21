import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("MouseDrag-v0_test.csv", header=None)
df2 = pd.read_csv("dropdown_ratio_1.csv")

df.columns = ['value']
df['moving_avg'] = df['value'].rolling(window=5).mean()
plt.figure(figsize=(10, 5))
plt.plot(df['moving_avg'], label='Moving Average (window=5)', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Successful Drag')
plt.title("Successful Drags per Episode (Moving Average)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
###########
plt.scatter(df2['timesteps'], df2['ratio_success'], marker='o', alpha=0.2)
plt.xlabel('Timesteps')
plt.ylabel('Relative episode length')
plt.title('Relative episode length over Time')
plt.grid(True)
plt.show()