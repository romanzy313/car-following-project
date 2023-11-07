from pathlib import Path
import pickle
from typing import Any
import numpy as np
import pandas as pd
import zarr

from sklearn.preprocessing import StandardScaler


def get_scaler(
    dataset: str, cluster: int, root_path: str = "../out_segmented"
) -> StandardScaler:
    with open(f"{root_path}/{dataset}_{cluster}_scaler.pkl", "rb") as f:
        sc = pickle.load(f)
        return sc


def get_train_data(
    dataset: str,
    cluster: int,
    type: str = "features",
    root_path: str = "../out_segmented",
):
    """
    type is either features or labels
    """
    path = f"{root_path}/{dataset}_{cluster}_{type}.h5"

    return pd.read_hdf(path, key=type)


# def read_clustered_data(data_file: str):
#     raw_data = zarr.open(data_file, mode="r")
#     data = pd.DataFrame(
#         # raw_data
#         {
#             "delta_velocity": raw_data[:, 0],
#             "delta_position": raw_data[:, 1],
#             "v_follower": raw_data[:, 2],
#         }
#     )
#     return data


def read_data(cfpair, dataset, root_folder="../data/"):
    data_file = Path(root_folder + dataset + cfpair + ".zarr/").resolve()
    data: Any = zarr.open(data_file, mode="r")  # type: ignore
    indexrange = data.index_range[:]

    if cfpair == "AH":
        leadsize = data.lead_size[:]
        followsize = 4.85 * np.ones(len(leadsize))
    elif cfpair == "HA":
        followsize = data.follow_size[:]
        leadsize = 4.85 * np.ones(len(followsize))
    else:
        leadsize = data.lead_size[:]
        followsize = data.follow_size[:]

    case_ids = np.zeros(len(data.timestamp)).astype(int)
    leader_size = np.zeros(len(data.timestamp))
    follower_size = np.zeros(len(data.timestamp))
    for case_id in np.arange(len(indexrange)):
        start, end = indexrange[case_id]
        case_ids[start:end] = case_id
        leader_size[start:end] = leadsize[case_id]
        follower_size[start:end] = followsize[case_id]

    data = pd.DataFrame(
        {
            "case_id": case_ids.astype(int),
            "time": np.round(data.timestamp, 1),
            "x_leader": data.lead_centroid,
            "x_follower": data.follow_centroid,
            "v_leader": data.lead_velocity,
            "v_follower": data.follow_velocity,
            "a_leader": data.lead_acceleration,
            "a_follower": data.follow_acceleration,
            "l_leader": leader_size,
            "l_follower": follower_size,
        }
    )

    return data
