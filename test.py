import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np

cog = np.random.randint(0, 359, 10)
print(cog)
print(np.sin(cog))