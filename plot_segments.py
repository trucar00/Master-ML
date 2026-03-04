import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data/line_01_is_steaming.csv")

print(df.head())


for mmsi, d in df.groupby("mmsi"):
    fig, ax = plt.subplots(figsize=(10,8))
    for seg, dd in d.groupby("segment_id"):
        if dd["is_steaming"].iloc[0] == 0:
            #print(dd.head())
            ax.plot(dd["lon"], dd["lat"], linewidth=1, alpha=0.7)
    
    plt.show()