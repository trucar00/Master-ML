import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data/trawl_segments_with_predictions.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

for mmsi, d in df.groupby("mmsi"):
    d = d.sort_values(by="datetime")
    fig, ax = plt.subplots(figsize=(10,8))
    for seg, dd in d.groupby("segment_id"):
        print(dd.shape)
        if dd["is_fishing"].iloc[0]:
            ax.plot(dd["lon"], dd["lat"], color="red", linewidth=1, alpha=0.7)
        else:
            ax.plot(dd["lon"], dd["lat"], color="blue", linewidth=1, alpha=0.7)
    
    plt.show()

