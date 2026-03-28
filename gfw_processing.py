import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

def haversine(lat1, lon1, lat2, lon2, dt):
    R = 6371000 # Radius of the earth in meters

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1


    # apply formulae
    a = (pow(np.sin(dlat / 2), 2) +  
             np.cos(lat1) * np.cos(lat2) * pow(np.sin(dlon / 2), 2))
    
    c = 2 * np.arcsin(np.sqrt(a))

    dist = R * c
    speed = (dist/dt) * 1.94384 # Convert m/s to knots

    return dist, speed

def remove_invalid(df, min_cog=0, max_cog=360, min_speed=0, max_speed=30):
    print("Removing invalid rows")

    # Ensure numeric columns
    for col in ["course", "speed", "lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build a mask for valid values
    valid_mask = (
        df["course"].between(min_cog, max_cog, inclusive="both")
        & df["speed"].between(min_speed, max_speed, inclusive="both")
    )

    invalid_count = len(df) - valid_mask.sum()
    print(f"Removed {invalid_count:,} invalid rows")

    return df[valid_mask]

def remove_stationary(df, speed_threshold=0.4, min_duration="15min"):
    print("Removing stationary")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values(["mmsi", "datetime"])

    df["stationary"] = df["speed"] < speed_threshold

    # Group changes in stationary state PER MMSI
    df["grp"] = (
        df.groupby("mmsi")["stationary"]
          .apply(lambda s: (s != s.shift()).cumsum())
          .reset_index(level=0, drop=True)
    )

    drop_idx = []

    for (_, _), g in df[df["stationary"]].groupby(["mmsi", "grp"]):
        duration = g["datetime"].max() - g["datetime"].min()
        if duration >= pd.Timedelta(min_duration):
            drop_idx.append(g.index)

    if drop_idx:
        df = df.drop(np.concatenate(drop_idx))

    return df.drop(columns=["stationary", "grp"])

def extract_trajectories(df, time_threshold="120min"):
    df = df.sort_values(["mmsi", "datetime"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    df["dt"] = df.groupby("mmsi")["datetime"].diff().dt.total_seconds()
    tt = pd.Timedelta(time_threshold).total_seconds()

    df["traj_id"] = (df["dt"] > tt).groupby(df["mmsi"]).cumsum()
    df["trajectory_id"] = df["mmsi"].astype(str) + "-" + df["traj_id"].astype(str)

    return df.drop(columns=["dt", "traj_id"])

def remove_outlier_positions(df, max_speed = 20):
    print("Removing trajectory outliers (impossible jumps)")
    df = df.sort_values(["trajectory_id", "datetime"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s") # Convert Unix timestamp → datetime

    df["dt_fwd"] = df.groupby("trajectory_id")["datetime"].diff().shift(-1).dt.total_seconds()
    df["dt_bwd"] = df.groupby("trajectory_id")["datetime"].diff().dt.total_seconds()

    df["lat_prev"] = df.groupby("trajectory_id")["lat"].shift()
    df["lon_prev"] = df.groupby("trajectory_id")["lon"].shift()
    df["lat_next"] = df.groupby("trajectory_id")["lat"].shift(-1)
    df["lon_next"] = df.groupby("trajectory_id")["lon"].shift(-1)

    df["del_m_fwd"], df["speed_fwd"] = haversine(df["lat"], df["lon"], df["lat_next"], df["lon_next"], df["dt_fwd"])
    df["del_m_bwd"], df["speed_bwd"] = haversine(df["lat_prev"], df["lon_prev"], df["lat"], df["lon"], df["dt_bwd"])

    df["del_speed_fwd"] = df.groupby("trajectory_id")["speed"].diff().shift(-1)
    df["del_speed_bwd"] = df.groupby("trajectory_id")["speed"].diff()
    
    df["accel_fwd"] = (df["del_speed_fwd"] / 1.94384) / df["dt_fwd"] # clean up the ms to knots thing?
    df["accel_bwd"] = (df["del_speed_bwd"] / 1.94384) / df["dt_bwd"]


    jump_mask = (df["speed_bwd"] > max_speed) & (df["speed_fwd"] > max_speed)
    accel_mask = (df["accel_fwd"].abs() > 0.10) & (df["accel_bwd"].abs() > 0.10)

    outlier_mask = jump_mask | accel_mask

    df_filtered = df[~outlier_mask].drop(columns=[
        "dt_fwd", "dt_bwd",
        "lat_prev", "lon_prev", "lat_next", "lon_next",
        "del_m_fwd", "del_m_bwd", "del_speed_fwd", "del_speed_bwd", "speed_fwd", "speed_bwd", "accel_fwd", "accel_bwd"
    ])

    print(f"Removed {outlier_mask.sum():,} outlier points")

    return df_filtered

def remove_duplicate_timestamps(df):
    print("Removing duplicate timestamps per trajectory")

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    before = len(df)

    # keep only the row with highest speed for each (trajectory_id, timestamp)
    df = (
        df.sort_values(["mmsi", "datetime", "speed"], ascending=[True, True, False])
          .drop_duplicates(subset=["mmsi", "datetime"], keep="first")
    )

    removed = before - len(df)
    print(f"Removed {removed:,} duplicate-timestamp rows")
    return df

def remove_short_trajectories(df, traj_length=2):
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    durations = (
    df.groupby("trajectory_id")["datetime"]
      .agg(["min", "max"])
      .assign(duration=lambda x: x["max"] - x["min"])
    )

    valid_traj_ids = durations[durations["duration"] >= pd.Timedelta(hours=traj_length)].index
    df_filtered = df[df["trajectory_id"].isin(valid_traj_ids)]
    print("Original:", df["trajectory_id"].nunique())
    print("Filtered:", df_filtered["trajectory_id"].nunique())

    return df_filtered

def downsample(df, step):
    df = df.copy()

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values(["trajectory_id", "datetime"]).set_index("datetime")
    
    theta = np.deg2rad(df["course"].astype(float))
    df["cog_x"] = np.cos(theta)
    df["cog_y"] = np.sin(theta)

    def resample_and_interpolate(g):
        traj = g.name  # <-- this group's trajectory_id string

        g_res = g.resample(step, origin=g.index.min()).first()

        # Interpolate only continuous signals that exist
        interp_cols = [c for c in ["lon", "lat", "speed", "cog_x", "cog_y"] if c in g_res.columns]
        if interp_cols:
            g_res[interp_cols] = g_res[interp_cols].interpolate("time", limit_area="inside")
        
        r = np.hypot(g_res["cog_x"], g_res["cog_y"])
        g_res["cog_x"] = g_res["cog_x"] / r
        g_res["cog_y"] = g_res["cog_y"] / r
        g_res["theta"] = np.arctan2(g_res["cog_y"], g_res["cog_x"])
        g_res["cog_interp"] = np.rad2deg(g_res["theta"]) % 360

        # Fill identifiers that exist
        id_cols = [c for c in ["mmsi", "source"] if c in g_res.columns]
        if id_cols:
            g_res[id_cols] = g_res[id_cols].ffill()

        g_res["is_fishing"] = (
                g["is_fishing"]
                  .resample(step, origin=g.index.min())
                  .nearest()
        )

        # Re-add trajectory_id as a string column
        g_res["trajectory_id"] = traj

        return g_res

    resampled = (
        df.groupby("trajectory_id", group_keys=False)
          .apply(resample_and_interpolate)
          .reset_index()
    )

    # Ensure trajectory_id is string dtype
    resampled["trajectory_id"] = resampled["trajectory_id"].astype("string")
    resampled["mmsi"] = resampled["mmsi"].astype("int64")

    resampled = resampled.drop(columns=["timestamp", "distance_from_shore", "distance_from_port", "course", "cog_x", "cog_y", "theta"])
    return resampled

def downsample2(df, step):
    df = df.copy()

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values(["trajectory_id", "datetime"]).set_index("datetime")
    
    theta = np.deg2rad(df["course"].astype(float))
    df["cog_x"] = np.cos(theta)
    df["cog_y"] = np.sin(theta)

    def resample_and_interpolate(g):
        traj = g.name  # <-- this group's trajectory_id string

         # regular grid from first to last timestamp of this trajectory
        regular_index = pd.date_range(
            start=g.index.min(),
            end=g.index.max(),
            freq=step
        )

        # union original timestamps + target grid, so interpolation uses real points
        g_union = g.reindex(g.index.union(regular_index)).sort_index()

        # linear spatial interpolation in time
        interp_cols = [c for c in ["lon", "lat"] if c in g_union.columns]
        g_union[interp_cols] = g_union[interp_cols].interpolate(
            method="time",
            limit_area="inside"
        )

        # optional interpolation of speed and course representation
        # speed is not essential here since you later recompute speed from position,
        # but keeping it can be useful for diagnostics
        extra_interp_cols = [c for c in ["speed", "cog_x", "cog_y"] if c in g_union.columns]
        g_union[extra_interp_cols] = g_union[extra_interp_cols].interpolate(
            method="time",
            limit_area="inside"
        )

        # keep only the regular timestamps
        g_res = g_union.loc[regular_index].copy()
        
        r = np.hypot(g_res["cog_x"], g_res["cog_y"])
        g_res["cog_x"] = g_res["cog_x"] / r
        g_res["cog_y"] = g_res["cog_y"] / r
        g_res["theta"] = np.arctan2(g_res["cog_y"], g_res["cog_x"])
        g_res["cog_interp"] = np.rad2deg(g_res["theta"]) % 360

        # Fill identifiers that exist
        id_cols = [c for c in ["mmsi", "source"] if c in g_res.columns]
        if id_cols:
            g_res[id_cols] = g_res[id_cols].ffill()

       
        lab = g[["is_fishing"]].copy()
        lab = lab[~lab.index.duplicated(keep="first")]
        g_res["is_fishing"] = (
            lab.reindex(regular_index, method="nearest")["is_fishing"].values
        )

        # Re-add trajectory_id as a string column
        g_res["trajectory_id"] = traj

        return g_res

    resampled = (
        df.groupby("trajectory_id", group_keys=False)
          .apply(resample_and_interpolate)
          .reset_index()
          .rename(columns={"index": "datetime"})
    )

    # Ensure trajectory_id is string dtype
    resampled["trajectory_id"] = resampled["trajectory_id"].astype("string")
    resampled["mmsi"] = resampled["mmsi"].astype("int64")

    resampled = resampled.drop(columns=["timestamp", "distance_from_shore", "distance_from_port", "course", "cog_x", "cog_y", "theta"])
    return resampled

def reindex_trajectory_ids(df):
    print("Reindexing trajectory IDs")

    df = df.sort_values(["mmsi", "datetime"])

    out = []
    for mmsi, g in df.groupby("mmsi", sort=False):
        # extract numeric suffix from old trajectory_id
        old = g["trajectory_id"].astype(str)
        old_num = old.str.rsplit("-", n=1).str[-1].astype(int)

        # get unique old IDs in numeric order
        order = (
            pd.DataFrame({"old": old, "num": old_num})
              .drop_duplicates("old")
              .sort_values("num")["old"]
              .tolist()
        )

        mapping = {oid: i for i, oid in enumerate(order)}
        newnum = old.map(mapping)

        g = g.copy()
        g["trajectory_id"] = g["mmsi"].astype(str) + "-" + newnum.astype(int).astype(str)
        out.append(g)

    return pd.concat(out, ignore_index=True)

def main():
    df = pd.read_csv("Data/gfw/purse_seines.csv")
    print(df.shape)
    fishing = df.loc[df["is_fishing"] >= 0.5]
    print(fishing.shape)
    #print(df_full["mmsi"].nunique()) # 49

    #first_20_mmsi = df["mmsi"].drop_duplicates().head(5)

    # Keep only those vessels
    #df = df[df["mmsi"].isin(first_20_mmsi)].copy()

    df["mmsi"] = df["mmsi"].astype("int64")

    # Cleaning:
    df = df.loc[df["distance_from_shore"] >= 10000]
    df = remove_invalid(df)
    df = remove_duplicate_timestamps(df)
    df = extract_trajectories(df)
    df = remove_outlier_positions(df)
    df = remove_short_trajectories(df) # Removes all trajectories shorter than 2h
    df = reindex_trajectory_ids(df)
    print(df.shape)
    fishing = df.loc[df["is_fishing"] >= 0.5]
    print(fishing.shape)
    df = downsample2(df, step="10min")
    fishing = df.loc[df["is_fishing"] >= 0.5]
    print(df.head())
    #print(fishing.shape)

    df.to_csv("purse_seines_gfw_processed.csv", index=False)

    # Keep only trajectory_ids that contain fishing
    traj_with_fishing = (
        df.loc[df["is_fishing"] >= 0.5, "trajectory_id"]
        .unique()
    )

    df_fishing_traj = df[df["trajectory_id"].isin(traj_with_fishing)]

    for mmsi, d in df_fishing_traj.groupby("trajectory_id"):
        fig, ax = plt.subplots(figsize=(10,8))
        d = d.sort_values("datetime")
        #print(d["datetime"].min(), d["datetime"].max())
        
       # Split fishing vs non-fishing
        fishing = d.loc[d["is_fishing"] >= 0.5]
        non_fishing = d.loc[d["is_fishing"] < 0.5]


        # Plot non-fishing in blue
        ax.scatter(non_fishing["lon"], non_fishing["lat"],
                s=2, color="blue", label="Non-fishing")

        # Plot fishing in red
        ax.scatter(fishing["lon"], fishing["lat"], s=4, color="red", label="Fishing")

        plt.show()

if __name__ == "__main__":
    main()
