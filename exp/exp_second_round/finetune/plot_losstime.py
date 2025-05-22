import pandas as pd
import matplotlib.pyplot as plt

# Load data
path_time = 'exp/exp_second_round/finetune/time.csv'
path_loss = 'exp/exp_second_round/finetune/loss.csv'
time_df = pd.read_csv(path_time).iloc[:, :2]
loss_df = pd.read_csv(path_loss).iloc[:, :2]

# Merge
merge_df = pd.merge(time_df, loss_df, on='Step')
merge_df['whole-voice-23 - time'] = merge_df['whole-voice-23 - time'].round(2)

# Plotting
plt.figure(figsize=(12, 6))

# Line plot
plt.plot(merge_df['Step'], merge_df['whole-voice-23 - mmd'], label='MMD', color='orange')

# Tick labels (Step + Time)
N = len(merge_df) // 10
tick_locs = merge_df['Step'][::N]
tick_labels = [
    f"Step: {s}, Time: {t}s"
    for s, t in zip(merge_df['Step'][::N], merge_df['whole-voice-23 - time'][::N])
]
plt.xticks(ticks=tick_locs, labels=tick_labels, rotation=20, ha='right', fontsize=18)

# Axis labels and title
plt.xlabel('Step [-], Time [Seconds]', fontsize=18)
plt.ylabel('MMD Loss [-]', fontsize=18)
# plt.title('Loss over Time', fontsize=16)
plt.legend(fontsize=18)
plt.grid(True)

# Annotate custom point
plt.scatter(x=-2.221, y=0.0198, color='red', zorder=5)
plt.annotate(
    'Our Method (MMD: 0.0198, Time: 0.021s)',
    xy=(0.021, 0.0198),
    xytext=(10, 20),
    textcoords='offset points',
    arrowprops=dict(arrowstyle='->', color='red'),
    fontsize=18,
    color='red'
)

# Draw horizontal line at MMD = 0.0198
plt.axhline(y=0.0198, color='gray', linestyle='--', linewidth=2)
plt.text(
    merge_df['Step'].max() * 0.98, 0.0198 + 0.001,
    'MMD = 0.0198',
    ha='right', va='bottom',
    fontsize=18, color='gray'
)

plt.tight_layout()
plt.savefig('exp/exp_second_round/finetune/loss_time_labeled.png')
plt.close()
