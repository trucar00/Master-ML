import pandas as pd
import matplotlib.pyplot as plt

def check_predicted(file):
    df = pd.read_csv(file)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df_fish = df.loc[df["is_fishing"] == 1].copy()
    df_steam = df.loc[df["is_fishing"] == 0].copy()

    print("Nr of unique segments: ", df_fish["segment_id"].nunique())
    print(df.shape)
    fig, ax = plt.subplots(figsize=(10,8))
    for mmsi, d in df.groupby("mmsi"):
        d = d.sort_values(by="datetime")
        
        for seg, dd in d.groupby("segment_id"):
            
            if dd["is_fishing"].iloc[0]:
                ax.plot(dd["lon"], dd["lat"], color="red", linewidth=1, alpha=0.7)
            else:
                ax.plot(dd["lon"], dd["lat"], color="blue", linewidth=1, alpha=0.7)
        
    plt.show()

def check(file):
    df = pd.read_csv(file)
    
    df["datetime"] = pd.to_datetime(df["datetime"])

    for mmsi, d in df.groupby("mmsi"):
        d = d.sort_values(by="datetime")
        fig, ax = plt.subplots(figsize=(10,8))
        for seg, dd in d.groupby("segment_id"):
            ax.plot(dd["lon"], dd["lat"], color="red", linewidth=1, alpha=0.7)
        
        plt.show()


check_predicted("../per_gear/t_new_segments_with_predictions.csv")
#check("Data/line_segments_no_label.csv")

# Classifies quite nice. Can try to create feature dataset with trawlers and longliners. But need to scale up from only january.
# Compare rule based and CNN for line/trawl.
