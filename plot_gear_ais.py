import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data/gear_specific/trawl_jan_2024.csv")
df_clean = pd.read_csv("Data/gear_specific/trawl_clean_downsampled.csv")

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
df_clean["date_time_utc"] = pd.to_datetime(df_clean["date_time_utc"])

first_20_mmsi = df_clean["mmsi"].drop_duplicates().head(10)

# Keep only those vessels
df_small = df_clean[df_clean["mmsi"].isin(first_20_mmsi)].copy()

for mmsi, d in df_small.groupby("mmsi"):
    fig, ax = plt.subplots(figsize=(10,8))
    nrTraj = d["trajectory_id"].nunique()
    for traj, dd in d.groupby("trajectory_id"):
        dd = dd.sort_values("date_time_utc")
        ax.plot(dd["lon"], dd["lat"], linewidth=1, alpha=0.7)
    plt.title(f"{mmsi} with {nrTraj} trajectories.")
    plt.show()

""" fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# --- Left plot: original data ---
for mmsi, d in df.groupby("mmsi"):
    d = d.sort_values("date_time_utc")
    axs[0].plot(d["lon"], d["lat"], linewidth=1, alpha=0.7)

axs[0].set_title("Original data")
axs[0].set_xlabel("Longitude")
axs[0].set_ylabel("Latitude")

# --- Right plot: cleaned data ---
for mmsi, d in df_clean.groupby("mmsi"):
    d = d.sort_values("date_time_utc")
    axs[1].plot(d["lon"], d["lat"], linewidth=1, alpha=0.7)

axs[1].set_title("Cleaned data")
axs[1].set_xlabel("Longitude")
axs[1].set_ylabel("Latitude")

plt.tight_layout()
plt.show() """