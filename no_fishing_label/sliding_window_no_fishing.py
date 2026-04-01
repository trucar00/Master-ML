import pandas as pd
import numpy as np
from tqdm import tqdm

# READY FOR CREATING SEGMENTS
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

df = pd.read_csv("Data/feats_trawl_no_label.csv")
print(df.head())
df["datetime"] = pd.to_datetime(df["datetime"])
df["traj_num"] = df["trajectory_id"].astype(str).str.rsplit("-", n=1).str[-1].astype(int)

df = df.sort_values(["mmsi", "traj_num", "datetime"])

nr_points = 11
slide = 5


all_feature_dfs = []
segment_id = 0  # global counter

for traj, d in tqdm(df.groupby("trajectory_id", sort=False)):
    d = d.sort_values("datetime").reset_index(drop=True)
    start_idx = 0
    end_idx = nr_points

    while end_idx <= len(d):
        segment = d.iloc[start_idx:end_idx]

       
        feature_df = pd.DataFrame() # must concat all segment dfs
        feature_df = segment.drop(columns=["speed", "cog_interp", "dt", "traj_num"]).copy()

        lon = segment["lon"].values
        lat = segment["lat"].values
        seg_length = segment["dist_to_prev"].sum()
        net_disp = haversine(lat[0], lon[0], lat[-1], lon[-1])
        straightness = net_disp / seg_length if seg_length > 0 else 0
        feature_df["straightness"] = straightness

        feature_df["mean_speed"] = segment["speed_calc_ms"].mean()
        feature_df["std_speed"] = segment["speed_calc_ms"].std()

        feature_df["segment_id"] = segment_id
        all_feature_dfs.append(feature_df)
        segment_id += 1

        start_idx += slide
        end_idx += slide

features_all = pd.concat(all_feature_dfs, ignore_index=True)
print(features_all.shape) # fishing + steaming segments * 11 

features_all.to_csv("Data/t_new_segments_no_label.csv", index=False)