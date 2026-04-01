import pandas as pd
import numpy as np

df = pd.read_csv("feats_segment_level_longlines.csv")

segments = []
labels = []
meta = []
groups = []

for seg_id, seg_df in df.groupby("segment_id"):
    seg_df = seg_df.sort_values("datetime") 

    X_seg = seg_df.drop(columns=["mmsi", "datetime", "trajectory_id", "segment_id", "fishing"]).values
    
    
    y_seg = seg_df["fishing"].iloc[0]

    traj_id = seg_df["trajectory_id"].iloc[0]
    mmsi = seg_df["mmsi"].iloc[0]

    segments.append(X_seg)
    labels.append(y_seg)
    groups.append(mmsi) # or traj_id

    meta.append({
        "segment_id": seg_id,
        "trajectory_id": seg_df["trajectory_id"].iloc[0],
        "mmsi": seg_df["mmsi"].iloc[0],
        "start_time": seg_df["datetime"].iloc[0],
        "end_time": seg_df["datetime"].iloc[-1]
    })

X = np.array(segments)   # (N, 11, features)
y = np.array(labels)
groups = np.array(groups)
meta = pd.DataFrame(meta) # index mathes, so if y[0] is predicted as fishibg, i can look up meta[0] to see what trajectory it is. 
np.save("datasets/X_line", X)
np.save("datasets/y_line", y)
np.save("datasets/meta_line", meta)
np.save("datasets/groups_line", groups)

print(X.shape, y.shape)