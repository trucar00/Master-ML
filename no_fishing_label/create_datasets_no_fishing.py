import pandas as pd
import numpy as np

df = pd.read_csv("Data/not_segments_no_label.csv")

segments = []
meta = []

for seg_id, seg_df in df.groupby("segment_id"):
    seg_df = seg_df.sort_values("datetime") 

    X_seg = seg_df.drop(columns=["mmsi", "datetime", "lon", "lat", "trajectory_id", "callsign", "segment_id"]).values

    traj_id = seg_df["trajectory_id"].iloc[0]

    segments.append(X_seg)

    meta.append({
        "segment_id": seg_id,
        "trajectory_id": seg_df["trajectory_id"].iloc[0],
        "mmsi": seg_df["mmsi"].iloc[0],
        "start_time": seg_df["datetime"].iloc[0],
        "end_time": seg_df["datetime"].iloc[-1]
    })

X = np.array(segments)   # (N, 11, features)
meta = pd.DataFrame(meta) # index mathes, so if y[0] is predicted as fishibg, i can look up meta[0] to see what trajectory it is. 
np.save("Data/datasets/X_not_no_label", X)
np.save("Data/datasets/meta_not_no_label", meta)

print(X.shape)