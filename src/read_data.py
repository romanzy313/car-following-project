from pathlib import Path
import numpy as np
import pandas as pd
import zarr


def read_data(cfpair, dataset, root_folder="../data/"):
    # regimes_file = Path(
    #     root_folder + "regimes_list_" + cfpair + "_" + dataset + ".csv"
    # ).resolve()
    data_file = Path(root_folder + dataset + cfpair + ".zarr/").resolve()

    # regimes_list = pd.read_csv(regimes_file, index_col=0)
    data: any = zarr.open(data_file, mode="r")  # type: ignore
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
