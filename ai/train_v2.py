import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from read_data import read_data

original_data_path = "../data"
separated_data_path = "../out_cluster"
np.random.seed()  # Set a random seed


# Read data
def read_data_wrapper(setname, type):
    # data = pd.read_hdf(
    #     f"{original_data_path}/{type}{setname}.zarr"
    # )  # fill in your path and file name
    data = read_data(setname, type)
    return data[["case_id", "time", "x_leader", "x_follower", "v_leader", "v_follower"]]


data_HA = read_data_wrapper("HA", "train")
data_HH = read_data_wrapper("HH", "train")


def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(0, len(data) - n_steps_in - n_steps_out + 5):
        seq_x = data[i : i + n_steps_in]
        seq_y = data[i + n_steps_in : i + n_steps_in + n_steps_out]
        if seq_x.shape[0] == n_steps_in and seq_y.shape[0] == n_steps_out:
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)


# Segment data to make 30 timesteps input and 10 timesteps output
def segment_data(data: pd.DataFrame):
    data["delta_velocity"] = data["v_follower"] - data["v_leader"]
    data["delta_position"] = data["x_leader"] - data["x_follower"]
    data = data.sort_values(by=["case_id", "time"]).set_index("case_id")
    features = []
    labels = []
    idx = 0

    print("data")
    print(data.head(10))
    for case_id in tqdm(data.index.unique()):
        df: pd.DataFrame = data.loc[case_id]
        future_idx_end = np.arange(
            40, len(df), 40
        )  # I create it without overlapping, do that if the data amount is not enough or as you wish

        # future_idx_end = np.concatenate(
        #     (future_idx_end, future_idx_end[1:] - 15)
        # )  # make 10 timesteps overlapping
        future_idx_start = future_idx_end - 10
        history_idx_end = future_idx_start
        history_idx_start = history_idx_end - 30

        print("history startto end", history_idx_start, history_idx_end)

        feature = df.iloc[history_idx_start:history_idx_end][
            ["time", "delta_velocity", "delta_position", "v_follower"]
        ]
        print("feature")
        print(feature.head(3))
        feature["idx"] = idx
        label = df.iloc[future_idx_start:future_idx_end]["time", "v_follower"]
        label["idx"] = idx
        features.append(feature)
        labels.append(label)
    features = pd.concat(features).reset_index()
    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    # But do not standardize labels
    labels = pd.concat(labels).reset_index()
    return features, labels


features_HA, labels_HA = segment_data(data_HA)
features_HA.to_hdf(
    separated_data_path + "features_HA.h5", key="features"
)  # or save in other format you are familiar with # type: ignore
labels_HA.to_hdf(separated_data_path + "labels_HA.h5", key="labels")
features_HH, labels_HH = segment_data(data_HH)
features_HH.to_hdf(separated_data_path + "features_HH.h5", key="features")  # type: ignore
labels_HH.to_hdf(separated_data_path + "labels_HH.h5", key="labels")

# Read features and labels in local
features_HA = pd.read_hdf(separated_data_path + "features_HA.h5", key="features")
labels_HA = pd.read_hdf(separated_data_path + "labels_HA.h5", key="labels")
features_HH = pd.read_hdf(separated_data_path + "features_HH.h5", key="features")
labels_HH = pd.read_hdf(separated_data_path + "labels_HH.h5", key="labels")


# # Split data into training, validation, test set as you wish
# train_features_HA =
# train_labels_HA = # make sure the random choice of features and labels are the same!
# train_features_HH =
# train_labels_HH =
# # ... and so on


# Create dataloader function
class CreateDataset:
    def __init__(self, features, labels):
        self.labels = labels.sort_values(["idx", "time"]).set_index("idx")
        self.features = features.sort_values(["idx", "time"]).set_index("idx")

    def __len__(self):
        return self.labels.index.nunique()

    def __getitem__(self, idx):
        # idx is the index of items in the data and labels
        history = self.features.loc[idx][
            ["delta_velocity", "delta_position", "v_follower"]
        ].values
        history = torch.from_numpy(history).float()
        future = self.labels[idx]["v_follower"].values
        future = torch.from_numpy(future).float()
        return history, future


# Create dataloader to be used
train_dataloader_HA = DataLoader(
    CreateDataset(train_features_HA, train_labels_HA), batch_size=64, shuffle=True
)  # batch_size can also be 128
# ... the same for others


# Test if the dataloader works
history, future = next(iter(train_dataloader_HA))
print(f"Feature batch shape: {history.size()}")
print(f"Labels batch shape: {future.size()}")
