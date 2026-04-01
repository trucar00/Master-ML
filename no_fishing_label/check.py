import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

X = np.load("Data/datasets/X_all.npy")
y = np.load("Data/datasets/y_all.npy")
groups = np.load("Data/datasets/groups_all.npy")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

split_df = pd.DataFrame({
    "group": groups.astype(str)
})

# First split: gear vs rest
g1 = split_df["group"].str.split("_", n=1, expand=True)
print(g1)
split_df["gear"] = g1[0]
rest = g1[1]

# Second split: mmsi vs traj
g2 = rest.str.split("-", n=1, expand=True)
split_df["mmsi"] = g2[0]
split_df["traj_num"] = g2[1]

vessel_groups = (g1[0] + "_" + g2[0]).values
print(vessel_groups)

train_idx, test_idx = next(gss.split(X, y, groups=vessel_groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]

# Assign train/test
split_df["set"] = "unused"
split_df.loc[train_idx, "set"] = "train"
split_df.loc[test_idx, "set"] = "test"

summary_vessels = split_df.groupby(["set", "gear"])["mmsi"].nunique()
print(summary_vessels)

purse_train = split_df[(split_df["set"] == "train") & (split_df["gear"] == "p")]["mmsi"].nunique()
purse_test = split_df[(split_df["set"] == "test") & (split_df["gear"] == "p")]["mmsi"].nunique()

print("Purse seiners in train:", purse_train)
print("Purse seiners in test:", purse_test)

train_vessels = set(vessel_groups[train_idx])
test_vessels = set(vessel_groups[test_idx])

print("Overlap:", len(train_vessels & test_vessels))