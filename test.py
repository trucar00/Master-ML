import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

mmsi = [257006840]

table = pq.read_table(
    "Data/january.parquet", #01clean2
    columns=["mmsi", "callsign", "date_time_utc", "lon", "lat"],
    filters=[("mmsi", "in", mmsi)]
)

df_ais = table.to_pandas()
print(df_ais.shape)

#df_ais.to_csv("Data/simple_not.csv", index=False)

plt.plot(df_ais["lon"], df_ais["lat"])
plt.show()