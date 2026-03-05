import pandas as pd
import numpy  as np

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

def accel(s1, s2, dt):
    return (s2 - s1) / dt

def jerk(a1, a2, dt):
    return (a2 - a1) / dt

def dcourse(cog1, cog2, dt):
    return (cog2 - cog1) / dt


if __name__ == "__main__":
    print("yo")