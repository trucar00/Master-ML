import pandas as pd
import feature_funcs as ff
import numpy as np
from tqdm import tqdm

# READY FOR CREATING SEGMENTS

df = pd.read_csv("feats_traj_level_trawl.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df["traj_num"] = df["trajectory_id"].astype(str).str.rsplit("-", n=1).str[-1].astype(int)

df = df.sort_values(["mmsi", "traj_num", "datetime"])

nr_points = 11
slide = 5
min_labeled = 6

fish_segs = 0
steam_segs = 0
no_label = 0

all_feature_dfs = []
segment_id = 0  # global counter

for traj, d in tqdm(df.groupby("trajectory_id", sort=False)):
    d = d.sort_values("datetime").reset_index(drop=True)
    start_idx = 0
    end_idx = nr_points

    while end_idx <= len(d):
        segment = d.iloc[start_idx:end_idx]
        labeled = segment[segment["is_fishing"] != -1]
        n_labeled = len(labeled)

        if n_labeled < min_labeled:
            no_label += 1
        else:
            feature_df = pd.DataFrame() # must concat all segment dfs
            feature_df = segment.drop(columns=["speed", "cog_interp", "dt", "lat", "lon", "source", "is_fishing", "traj_num"]).copy()

            # Generating segmentvise features
            lon = segment["lon"].values
            lat = segment["lat"].values
            seg_length = segment["dist_to_prev"].sum()
            net_disp = ff.haversine(lat[0], lon[0], lat[-1], lon[-1])
            straightness = net_disp / seg_length if seg_length > 0 else 0
            feature_df["straightness"] = straightness

            feature_df["mean_speed"] = segment["speed_calc_ms"].mean()
            feature_df["std_speed"] = segment["speed_calc_ms"].std()

            vote = (labeled["is_fishing"] > 0.5).sum()
            if vote >= (n_labeled/2):
                fish_segs += 1
                feature_df["fishing"] = 1
            else:
                steam_segs += 1
                feature_df["fishing"] = 0
            
            feature_df["segment_id"] = segment_id
            all_feature_dfs.append(feature_df)
            segment_id += 1

        start_idx += slide
        end_idx += slide

features_all = pd.concat(all_feature_dfs, ignore_index=True)
print(features_all.shape) # fishing + steaming segments * 11 
print(features_all.head())
print("Fishing segments: ", fish_segs, " Steaming segs: ", steam_segs, " No label: ", no_label)
features_all.to_csv("new_feats_segment_trawl.csv", index=False)