import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def remove_speedy_msgs(df, speed_threshold=8, min_duration="10min"):
    print("Removing hsm")

    df = df.copy()
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["mmsi", "date_time_utc"])

    df["high_speed"] = df["speed"] >= speed_threshold

    # Group changes in stationary state PER MMSI
    df["grp"] = (
        df.groupby("mmsi")["high_speed"]
          .apply(lambda s: (s != s.shift()).cumsum())
          .reset_index(level=0, drop=True)
    )

    drop_idx = []

    for (_, _), g in df[df["high_speed"]].groupby(["mmsi", "grp"]):
        duration = g["date_time_utc"].max() - g["date_time_utc"].min()
        if duration >= pd.Timedelta(min_duration):
            drop_idx.append(g.index)

    if drop_idx:
        df = df.drop(np.concatenate(drop_idx))

    return df.drop(columns=["high_speed", "grp"])

def extract_trajectories(df, time_threshold="60min"):
    df = df.sort_values(["mmsi", "date_time_utc"])
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    df["dt"] = df.groupby("mmsi")["date_time_utc"].diff().dt.total_seconds()
    tt = pd.Timedelta(time_threshold).total_seconds()

    df["traj_id"] = (df["dt"] > tt).groupby(df["mmsi"]).cumsum()
    df["trajectory_id"] = df["mmsi"].astype(str) + "-" + df["traj_id"].astype(str)

    return df.drop(columns=["dt"])

def remove_trajectories_few_instances(df, min_instances=50):
    print(f"Removing trajectories with fewer than {min_instances} messages")

    counts = df["trajectory_id"].value_counts()
    valid_traj = counts[counts >= min_instances].index
    df_filtered = df[df["trajectory_id"].isin(valid_traj)]

    removed = len(counts) - len(valid_traj)
    print(f"Removed {removed} trajectories")

    return df_filtered

def remove_duplicate_timestamps(df):
    print("Removing duplicate timestamps per trajectory")

    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    before = len(df)

    # keep only the row with highest speed for each (trajectory_id, timestamp)
    df = (
        df.sort_values(["trajectory_id", "date_time_utc", "speed"], ascending=[True, True, False])
          .drop_duplicates(subset=["trajectory_id", "date_time_utc"], keep="first")
    )

    removed = before - len(df)
    print(f"Removed {removed:,} duplicate-timestamp rows")
    return df

df = pd.read_csv("Data/gear_specific/line_jan_2024.csv")

print(df.shape)
df_no_speed = remove_speedy_msgs(df)
print(df_no_speed.shape)
df_split_traj = extract_trajectories(df_no_speed)
print(df_split_traj.head())
df_split_traj = remove_duplicate_timestamps(df_split_traj)
print(df_split_traj.shape)
df_clean = remove_trajectories_few_instances(df_split_traj)
print("After clean: ", df_clean.shape)

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

for mmsi, d in df.groupby("mmsi"):
    d["date_time_utc"] = pd.to_datetime(d["date_time_utc"])
    d = d.sort_values(by="date_time_utc")
    ax[0].scatter(d["lon"], d["lat"], s=1)

for mmsi, d in df_clean.groupby("mmsi"):
    d["date_time_utc"] = pd.to_datetime(d["date_time_utc"])
    d = d.sort_values(by="date_time_utc")
    ax[1].scatter(d["lon"], d["lat"], s=1)

plt.tight_layout()
plt.show()