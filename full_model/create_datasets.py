import pandas as pd
import numpy as np

df = pd.read_csv("Data/feats_segment_level_line_trawl.csv")

segments = []
labels = []
meta = []
groups = []

trawl_seg = 0
line_seg = 0

for seg_id, seg_df in df.groupby("segment_id"):
    seg_df = seg_df.sort_values("datetime") 
    gear = seg_df["gear"].iloc[0]

    X_seg = seg_df.drop(columns=["gear", "mmsi", "datetime", "trajectory_id", "trajectory_uid", "segment_id", "fishing"])

    X_seg_np = X_seg.values
    
    
    y_seg = seg_df["fishing"].iloc[0]
    mmsi = seg_df["mmsi"].iloc[0]

    segments.append(X_seg_np)
    labels.append(y_seg)
    groups.append(gear + "-" + str(mmsi))

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
np.save("Data/datasets/X_line_trawl", X)
np.save("Data/datasets/y_line_trawl", y)
np.save("Data/datasets/meta_line_trawl", meta)
np.save("Data/datasets/groups_line_trawl", groups)

print(X.shape, y.shape)
print("t: ", trawl_seg, " l: ", line_seg)