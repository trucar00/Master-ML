import pandas as pd
import numpy as np
from tqdm import tqdm

# READY FOR CREATING SEGMENTS

df = pd.read_csv("Data/feats_trawl_no_label.csv")
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

        feature_df["segment_id"] = segment_id
        all_feature_dfs.append(feature_df)
        segment_id += 1

        start_idx += slide
        end_idx += slide

features_all = pd.concat(all_feature_dfs, ignore_index=True)
print(features_all.shape) # fishing + steaming segments * 11 

features_all.to_csv("Data/trawl_segments_no_label.csv", index=False)