import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("no3.csv", header=None)
df2 = pd.read_csv("ratio3.csv")

df["episode"] = df.index + 1
df["success_click"] = df[0]

plt.scatter(df["episode"], df["success_click"], marker='o')

plt.title("Successful Drags per Episode")
plt.xlabel("Episode")
plt.ylabel("Successful Drags")

plt.grid(True)
plt.tight_layout()
plt.show()
###########
plt.scatter(df2['timesteps'], df2['ratio_success'], marker='o', alpha=0.2)
plt.xlabel('Timesteps')
plt.ylabel('Success Ratio')
plt.title('Success Ratio over Time')
plt.grid(True)
plt.show()