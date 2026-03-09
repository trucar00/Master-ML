import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np

cog = np.random.randint(0, 359, 10)
print(cog)
print(np.sin(cog))

def test_interpolate(df):
    

    df = df.set_index("time")

    # resample to 30 minutes
    df = df.resample("30min").interpolate("time")

    return df.reset_index()

def test_interpolate_cog(df):
    theta = np.deg2rad(df["cog"].astype(float))
    df["cog_x"] = np.cos(theta)
    df["cog_y"] = np.sin(theta)
    df = df.set_index("time")

    # resample to 30 minutes
    df = df.resample("30min").interpolate("time")

    r = np.hypot(df["cog_x"], df["cog_y"])
    df["cog_x"] = df["cog_x"] / r
    df["cog_y"] = df["cog_y"] / r

    df["theta"] = np.arctan2(df["cog_y"], df["cog_x"])
    df["cogdeg"] = np.rad2deg(df["theta"])
    df["cog2"] = np.rad2deg(df["theta"]) % 360

    df = df.drop(columns=["cog", "cog_x", "cog_y","theta"])
    return df.reset_index()

df = pd.DataFrame(columns=["time", "cog"])
df["time"] = pd.date_range("2018-01-01", periods=6, freq="h")
df["cog"] = [350, 10, 355, 20, 40, 358]
df["speed"] = [1, 3, 2, 6, 3, 4]
print(df)
#df["del_cog"] = df["cog"].diff()
df = test_interpolate_cog(df)
print(df)