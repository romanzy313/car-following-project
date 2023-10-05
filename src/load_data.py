import re
import zarr

# from typing import Any
from typing import TypedDict
import numpy as np
import pandas as pd


# need to load zar, and to extract it
# def load_data(following: str, leading: str, isTrainingData: bool):
#     # filename = isTrainingData ? "train" : "val"
#     filename = get_filename(following, leading, isTrainingData)

#     data: zarr.Group = zarr.open(filename, mode="r")  # type: ignore
#     print("given data info is", data.info)


class Data_Slice(TypedDict):
    timestamp: float

    id_follow: int
    x_follow: float
    v_follow: float
    a_follow: float

    id_lead: int
    x_lead: float
    v_lead: float
    a_lead: float


def load_all_data(dataset: str):
    result = []

    drivers = re.search("([A|H])([A|H])", dataset)
    if drivers == None:
        raise ValueError("bad dataset name")

    follow = drivers[1]
    lead = drivers[2]

    data: zarr.Group = zarr.open("data/" + dataset + ".zarr", mode="r")  # type: ignore
    max_range = len(data.index_range[:])
    # print(max_range)
    for n in range(0, max_range):
        start, end = data.index_range[n]
        timestamps = data.timestamp[start:end]

        id_follow = -1
        id_lead = -1
        if follow == "H":
            id_follow = data.id[0]
        if lead == "H":
            if follow == "H":
                id_lead = data.id[1]
            else:
                id_lead = data.id[0]

        x_follow = data.follow_centroid[start:end]
        v_follow = data.follow_velocity[start:end]
        a_follow = data.follow_acceleration[start:end]

        # get position, speed, and acceleration
        x_lead = data.lead_centroid[start:end]
        v_lead = data.lead_velocity[start:end]
        a_lead = data.lead_acceleration[start:end]

        # numpy method
        # combined = np.column_stack(
        #     [
        #         np.array(timestamps),
        #         np.array(x_follow),
        #         np.array(v_follow),
        #         np.array(a_follow),
        #         np.array(x_lead),
        #         np.array(v_lead),
        #         np.array(a_lead),
        #     ]
        # )
        # column_names = [
        #     "timestamp",
        #     "x_follow",
        #     "v_follow",
        #     "a_follow",
        #     "x_lead",
        #     "v_lead",
        #     "a_lead",
        # ]
        # df = pd.DataFrame(combined, columns=column_names)

        # pandas method
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "x_follow": x_follow,
                "v_follow": v_follow,
                "a_follow": a_follow,
                "x_lead": x_lead,
                "v_lead": v_lead,
                "a_lead": a_lead,
            }
        )
        # print(df)

        result.append(df)
    return result
