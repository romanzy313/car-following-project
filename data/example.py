from __future__ import annotations
import zarr

import numpy as np

# import pandas as pd


########################
"""
Explanation available: 
https://github.com/RomainLITUD/Car-Following-Dataset-HV-vs-AV
"""
########################

n = 10

# read dataset
data: zarr.Group = zarr.open("./trainHH.zarr", mode="r")  # type: ignore
start, end = data.index_range[n]

# get vehicle size
size_lead = 4.85  # this is for AV
size_lead = data.lead_size[n]  # this is for HV
size_follow = data.follow_size[n]

# get timestamps
timestamps = data.timestamp[start:end]

# get position, speed, and acceleration
x_lead = data.lead_centroid[start:end]
v_lead = data.lead_velocity[start:end]
a_lead = data.lead_acceleration[start:end]

x_follow = data.follow_centroid[start:end]
v_follow = data.follow_velocity[start:end]
a_follow = data.follow_acceleration[start:end]

# merged = [timestamps, x_follow, v_follow, a_follow]

print(data.info)
