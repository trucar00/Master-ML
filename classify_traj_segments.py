import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 # Radius of the earth in meters

    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    # convert to radians
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0

    # apply formulae
    a = (pow(np.sin(dLat / 2), 2) + 
         pow(np.sin(dLon / 2), 2) * 
             np.cos(lat1) * np.cos(lat2))
    
    c = 2 * np.arcsin(np.sqrt(a))

    dist = R * c

    return dist

PATH = "Data/gear_specific/line_clean_downsampled.csv"
TURN_THRESHOLD = 20  # degrees

df = pd.read_csv(PATH)

#first_20_mmsi = df_full["mmsi"].drop_duplicates().tail(5)

# Keep only those vessels
#df = df_full[df_full["mmsi"].isin(first_20_mmsi)].copy()

def angle_wrap(a):
    
    return (a + 180) % 360 - 180

df["del_cog"] = df.groupby("mmsi")["cog"].diff().apply(angle_wrap)
df["del_cog"] = df["del_cog"].fillna(0)
df["is_steaming"] = 0
df["segment_id"] = pd.NA

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
print(df["trajectory_id"].nunique())

window_length = pd.Timedelta(hours=3)

for traj_id, idx in tqdm(df.groupby("trajectory_id").groups.items()):
    wind_id = 0

    d = df.loc[idx].sort_values("date_time_utc")  # keep original index
    start = d["date_time_utc"].min()
    end   = d["date_time_utc"].max()
    current = start

    while current + window_length <= end:

        in_window = (d["date_time_utc"] >= current) & (d["date_time_utc"] < current + window_length)
        window_df_all = d.loc[in_window]  # includes NaNs

        # assign segment_id to ALL rows in the window (including NaNs)
        seg = f"{traj_id}-{wind_id}"
        df.loc[window_df_all.index, "segment_id"] = seg

        window_df_clean = window_df_all.dropna(subset=["lon", "lat", "speed", "cog"])
        if len(window_df_clean) >= 5:
            # --- your stats & score (unchanged) ---
            window_avg_speed = window_df_clean["speed"].mean()
            window_std_speed = window_df_clean["speed"].std()

            lon = window_df_clean["lon"].values
            lat = window_df_clean["lat"].values
            step_dist = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
            path_length = step_dist.sum()
            net_disp = haversine(lat[0], lon[0], lat[-1], lon[-1])
            straightness = net_disp / path_length if path_length > 0 else 0

            n_large_turns = np.sum(np.abs(window_df_clean["del_cog"]) > TURN_THRESHOLD)

            score = 0
            score += (window_avg_speed > 8)
            score += (window_std_speed < 2)
            score += (straightness > 0.8)
            score += (n_large_turns < 3)

            if score >= 3:
                df.loc[window_df_all.index, "is_steaming"] = 1

        wind_id += 1
        current += window_length


df.to_csv("Data/line_01_is_steaming.csv", index=False)
#steaming_df = pd.concat(steaming)
#fishing_df = pd.concat(fishing)


#df.to_csv("fishing01.csv")

print(df.head())

for mmsi, d in df.groupby("mmsi"):
    fig, ax = plt.subplots(figsize=(10,8))
    nrTraj = d["trajectory_id"].nunique()
    for traj, dd in d.groupby("trajectory_id"):
        dd = dd.sort_values("date_time_utc")
        lon_plot = dd["lon"].where(dd["is_steaming"] == 1)
        lat_plot = dd["lat"].where(dd["is_steaming"] == 1)

        ax.plot(lon_plot, lat_plot, linewidth=1, alpha=0.7, color="blue")

        lon_plot = dd["lon"].where(dd["is_steaming"] == 0)
        lat_plot = dd["lat"].where(dd["is_steaming"] == 0)

        ax.plot(lon_plot, lat_plot, linewidth=1, alpha=0.7, color="red")
    
    plt.title(f"{mmsi} with {nrTraj} trajectories.")
    plt.show()

# looks very ugly because we plot all with steaming = 1 -> creates gaps where fishing is "detected" plots between these gaps creating big jumps. Same for fishing
# need to divide into trajectories, steaming trajectory, fishing trajectory. then it will make more sense.
# then try for the trawling and autoline only datasets. see if we can remove all steaming trajectories. 