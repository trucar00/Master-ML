from pyproj import Transformer
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
import numpy as np

tf = Transformer.from_crs(4326, 3035, always_xy=True)

df = pd.read_csv("Data/simple_trawl.csv", usecols=["mmsi", "date_time_utc", "lon", "lat"])

df["x"], df["y"] = tf.transform(df["lon"], df["lat"])

cell = 20000  # meters per grid cell (choose your resolution)

# Choose an origin for the grid (common choices: min values, or a fixed reference)
x_min = df["x"].min()
y_min = df["y"].min()

# Integer bin indices
df["col"] = np.floor((df["x"] - x_min) / cell).astype(int)
df["row"] = np.floor((df["y"] - y_min) / cell).astype(int)

df["x_p"] = (df["col"] * cell) + x_min
df["y_p"] = (df["row"] * cell) + y_min

patch_pos = df.drop_duplicates(subset=["x_p", "y_p"])

fig, ax = plt.subplots(figsize=(7, 7))
for r in patch_pos.itertuples(index=False):
    ax.add_patch(Rectangle((r.x_p, r.y_p), cell, cell))

ax.plot(df["x"], df["y"], linewidth=1, color="red")

grid_spacing = 20000
label_spacing = 80000

ax.xaxis.set_major_locator(MultipleLocator(label_spacing, offset=x_min))
ax.yaxis.set_major_locator(MultipleLocator(label_spacing, offset=y_min))

ax.xaxis.set_minor_locator(MultipleLocator(grid_spacing, offset=x_min))
ax.yaxis.set_minor_locator(MultipleLocator(grid_spacing, offset=y_min))

ax.grid(which="minor", linewidth=0.4)
ax.grid(which="major", linewidth=1.0)

ax.set_xlabel("x [meters] (EPSG:3035)")
ax.set_ylabel("y [meters] (EPSG:3035)")
ax.set_title("Vessel trajectory with 20 km grid")

ax.set_aspect("equal", adjustable="box")
plt.show()
