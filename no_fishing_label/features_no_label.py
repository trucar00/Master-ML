import pandas as pd
import feature_funcs as ff
import numpy as np
from tqdm import tqdm

df = pd.read_csv("Data/gear_specific/trawl_clean_downsampled10min.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df["traj_num"] = df["trajectory_id"].astype(str).str.rsplit("-", n=1).str[-1].astype(int)

df = df.sort_values(["mmsi", "traj_num", "datetime"])

traj_level_feats = []

#first_traj_id = df["trajectory_id"].iloc[0]
#df = df[df["trajectory_id"] == first_traj_id].copy()

for traj, d in tqdm(df.groupby("trajectory_id", sort=False)):
    d = d.sort_values("datetime")
    dt = d["datetime"].diff().dt.total_seconds()
    d["dt"] = dt

    # Generate features on trajectory level

    # Distance between consecutive points
    lon1, lon2 = d["lon"].values[:-1], d["lon"].values[1:]
    lat1, lat2 = d["lat"].values[:-1], d["lat"].values[1:]
    dist = ff.haversine(lat1, lon1, lat2, lon2)
    dist = np.insert(dist, 0, np.nan)
    d["dist_to_prev"] = dist

    # Speed
    d["speed_calc_ms"] = d["dist_to_prev"] / d["dt"]

    # Acceleration
    d["accel"] = d["speed_calc_ms"].diff() / d["dt"]

    # Jerk
    d["jerk"] = d["accel"].diff() / d["dt"]

    # Derivative of course
    d["dcog"] = d["cog_interp"].diff().apply(ff.angle_wrap) / d["dt"]

    feature_cols = ["dist_to_prev", "speed_calc_ms", "accel", "jerk", "dcog"]
    d = d.dropna(subset=feature_cols)
    traj_level_feats.append(d)
    

df_with_feats = pd.concat(traj_level_feats)
df_with_feats.to_csv("Data/feats_trawl_no_label.csv", index=False)