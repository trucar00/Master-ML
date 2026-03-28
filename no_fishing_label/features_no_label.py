import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv("Data/gear_specific/not_clean_downsampled10min.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df["traj_num"] = df["trajectory_id"].astype(str).str.rsplit("-", n=1).str[-1].astype(int)

df = df.sort_values(["mmsi", "traj_num", "datetime"])


def angle_wrap(a):
    return (a + 180) % 360 - 180

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 # Radius of the earth in meters

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1


    # apply formulae
    a = (pow(np.sin(dlat / 2), 2) +  
             np.cos(lat1) * np.cos(lat2) * pow(np.sin(dlon / 2), 2))
    
    c = 2 * np.arcsin(np.sqrt(a))

    dist = R * c
    #speed = (dist/dt) * 1.94384 # Convert m/s to knots

    return dist #, speed

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
    dist = haversine(lat1, lon1, lat2, lon2)
    dist = np.insert(dist, 0, np.nan)
    d["dist_to_prev"] = dist

    # Speed
    d["speed_calc_ms"] = d["dist_to_prev"] / d["dt"]

    # Acceleration
    d["accel"] = d["speed_calc_ms"].diff() / d["dt"]

    # Jerk
    d["jerk"] = d["accel"].diff() / d["dt"]

    # Derivative of course
    d["dcog"] = d["cog_interp"].diff().apply(angle_wrap) / d["dt"]

    feature_cols = ["dist_to_prev", "speed_calc_ms", "accel", "jerk", "dcog"]
    d = d.dropna(subset=feature_cols)
    traj_level_feats.append(d)
    

df_with_feats = pd.concat(traj_level_feats)
df_with_feats.to_csv("Data/feats_traj_not_no_label.csv", index=False)