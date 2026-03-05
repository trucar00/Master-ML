import pandas as pd
import feature_funcs as ff

df = pd.read_csv("gfw_trawlers.csv")
df = df.drop(columns=["timestamp"])
df["datetime"] = pd.to_datetime(df["datetime"])
df["traj_num"] = df["trajectory_id"].astype(str).str.rsplit("-", n=1).str[-1].astype(int)

df = df.sort_values(["mmsi", "traj_num", "datetime"])

nr_points = 11
slide = 5
min_labeled = 6

fish_segs = 0
steam_segs = 0
no_label = 0

feature_df = pd.DataFrame(columns=["mmsi", "datetime", "dist_to_prev", "jerk", "speed",])
feature_df

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
            vote = (labeled["is_fishing"] > 0.5).sum()
            if vote >= (n_labeled/2):
                fish_segs += 1
            else:
                steam_segs += 1

        start_idx += slide
        end_idx += slide


print("Fishing segments: ", fish_segs, " Steaming segs: ", steam_segs, " No label: ", no_label)