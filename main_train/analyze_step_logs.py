import os
import json
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "20251107_pen3x3transfer1x1_num_1"

def analyze_step_rewards(version: str):
    STEP_DEBUG_DIR = os.path.join(f"../training_outputs/{version}/", "step_debug/")

    if not os.path.exists(STEP_DEBUG_DIR):
        print(f"Error: Directory not found: {STEP_DEBUG_DIR}")
        print("Please check the 'VERSION' variable.")
        return

    all_steps_data = []

    print(f"Scanning for log files in {STEP_DEBUG_DIR}...")

    for root, dirs, files in os.walk(STEP_DEBUG_DIR):
        for file in files:
            if file == "_step_log.json":
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, 'r') as f:
                        step_data = json.load(f)
                        all_steps_data.extend(step_data)
                except Exception as e:
                    print(f"Warning: Could not read or parse {json_path}. Error: {e}")

    if not all_steps_data:
        print("Error: No step log data was found.")
        return

    print(f"Successfully loaded {len(all_steps_data)} total steps from all episodes.")

    df = pd.DataFrame(all_steps_data)

    df = df.dropna(subset=['reward'])

    positive_reward_steps = df[df['reward'] > 0].shape[0]

    negative_reward_steps = df[df['reward'] <= 0].shape[0]

    total_steps = positive_reward_steps + negative_reward_steps

    print("\n--- Reward Analysis Statistics ---")
    print(f"  Total analyzed steps (reset steps excluded): {total_steps}")
    print(f"  Steps with Reward > 0: {positive_reward_steps}")
    print(f"  Steps with Reward <= 0: {negative_reward_steps}")

    if total_steps > 0:
        print(f"  Percentage of Positive Reward Steps: {positive_reward_steps / total_steps * 100:.2f}%")
        print(f"  Percentage of Negative/Zero Reward Steps: {negative_reward_steps / total_steps * 100:.2f}%")
    else:
        print("\nNo data available to plot.")
        return

    labels = [f'Positive Rewards (> 0)\n{positive_reward_steps} Steps',
              f'Negative/Zero Rewards (<= 0)\n{negative_reward_steps} Steps']
    counts = [positive_reward_steps, negative_reward_steps]

    plt.figure(figsize=(10, 7))
    bars = plt.bar(labels, counts, color=['#4CAF50', '#F44336'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01 * total_steps, int(yval),
                 ha='center', va='bottom', fontsize=12)

    plt.title(f'Step Reward Distribution for Version: {version}', fontsize=16)
    plt.ylabel('Number of Steps', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    analyze_step_rewards(VERSION)