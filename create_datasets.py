import pandas as pd
import numpy as np

df = pd.read_csv("Data/feats_traj_segments.csv")

segments = []
labels = []
meta = []

for seg_id, seg_df in df.groupby("segment_id"):
    seg_df = seg_df.sort_values("datetime") 

    X_seg = seg_df.drop(columns=["mmsi", "datetime", "trajectory_id", "segment_id", "fishing"]).values
    y_seg = seg_df["fishing"].iloc[0]

    segments.append(X_seg)
    labels.append(y_seg)

    meta.append({
        "segment_id": seg_id,
        "trajectory_id": seg_df["trajectory_id"].iloc[0],
        "mmsi": seg_df["mmsi"].iloc[0],
        "start_time": seg_df["datetime"].iloc[0],
        "end_time": seg_df["datetime"].iloc[-1]
    })

X = np.array(segments)   # (N, 11, features)
y = np.array(labels)
meta = pd.DataFrame(meta) # index mathes, so if y[0] is predicted as fishibg, i can look up meta[0] to see what trajectory it is. 
np.save("Data/datasets/X", X)
np.save("Data/datasets/y", y)
np.save("Data/datasets/meta", meta)

print(X.shape, y.shape)