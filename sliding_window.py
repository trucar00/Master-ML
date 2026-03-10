import pandas as pd
import feature_funcs as ff
import numpy as np

# READY FOR CREATING SEGMENTS

df = pd.read_csv("Data/gfw_trawlers_cog.csv")
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

for traj, d in df.groupby("trajectory_id", sort=False):
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
            feature_df["mmsi"] = segment["mmsi"]
            feature_df["trajectory_id"] = segment["trajectory_id"]
            feature_df["datetime"] = segment["datetime"]
            feature_df["lon"] = segment["lon"]
            feature_df["lat"] = segment["lat"]
            feature_df["cog_interp"] = segment["cog_interp"]
            feature_df["del_cog"] = segment["cog_interp"].diff().apply(ff.angle_wrap)
            
            lon1, lon2 = segment["lon"][:-1], segment["lon"][1:]
            lat1, lat2 = segment["lat"][:-1], segment["lat"][1:]
            dist = ff.haversine(lat1, lon1, lat2, lon2)
            dist = np.insert(dist, 0, np.nan)
            feature_df["dist_to_prev"] = dist

            dt = feature_df["datetime"].diff().dt.total_seconds()
            speed = ff.speed(dist, dt)
            feature_df["speed"] = speed
            #feature_df["accel"] = ff.accel() # we'll get two nans?
            #feature_df["jerk"]
            
            vote = (labeled["is_fishing"] > 0.5).sum()
            if vote >= (n_labeled/2):
                fish_segs += 1
                feature_df["fishing"] = 1
            else:
                steam_segs += 1
                feature_df["fishing"] = 0
            
            all_feature_dfs.append(feature_df)

        start_idx += slide
        end_idx += slide

features_all = pd.concat(all_feature_dfs)
print(features_all.shape) # fishing + steaming segments * 11 

print("Fishing segments: ", fish_segs, " Steaming segs: ", steam_segs, " No label: ", no_label)
features_all = features_all.drop(columns=["mmsi", "trajectory_id", "datetime"])
print(features_all.head())