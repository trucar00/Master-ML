import pandas as pd
import numpy  as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 # Radius of the earth in meters

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1


    # apply formulae
    a = (pow(np.sin(dlat / 2), 2) +  
             np.cos(lat1) * np.cos(lat2) * pow(np.sin(dlon / 2), 2))
    
    c = 2 * np.arcsin(np.sqrt(a))

    dist = R * c
    #speed = (dist/dt) * 1.94384 # Convert m/s to knots

    return dist #, speed

def speed(dist, dt):
    return (dist/dt) * 1.94384

def angle_wrap(a):
    return (a + 180) % 360 - 180

def accel(s1, s2, dt):
    return (s2 - s1) / dt

def jerk(a1, a2, dt):
    return (a2 - a1) / dt

def dcourse(cog1, cog2, dt):
    return (cog2 - cog1) / dt


if __name__ == "__main__":
    print("yo")
    df = pd.DataFrame(columns=["cog", "del_cog", "pos"])
    df["cog"] = [360, 10, 360, 20, 40, 358]
    df["del_cog"] = df["cog"].diff()
    df["del_cog"] = angle_wrap(df["del_cog"])
    #print(df.head())

    df["pos"] = [1, 2, 3, 4, 5, 6]
    print(df["pos"][:-1])
    print(df["pos"][1:])
    