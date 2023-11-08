# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from read_data import read_data
from read_data import get_scaler, get_train_data

original_data_path = "./data/"  # this is my path, change it to yours
# separated_data_path =

# %%
# # Read data
# def read_data(setname):
#     data = pd.read_hdf(original_data_path+"train"+setname+'.zarr', key='data') # fill in your path and file name
#     return data[['case_id','time','x_leader','x_follower','v_leader','v_follower']]
from read_data import read_data


def read_data_wrapper(setname, type):
    # data = pd.read_hdf(
    #     f"{original_data_path}/{type}{setname}.zarr"
    # )  # fill in your path and file name
    data = read_data(setname, type)
    data = data[["case_id", "time", "x_leader", "x_follower", "v_leader", "v_follower"]]
    features, labels = segment_data(data.loc[data["case_id"] < (1e5 + 500)])
    print("number of samples in HA:", labels["sample_id"].nunique())

    return features, labels


# data_HA = read_data('HA') # I guess this is HA_train or something in your PC
# data_HH = read_data('HH')

# features,labels = read_data_wrapper("HA", "train")


# %%
# data_HA.head()


# %%
# Segment data to make 30 timesteps input and 10 timesteps output
def segment_data(data):
    data = data.copy()
    data["delta_velocity"] = data["v_follower"] - data["v_leader"]
    data["delta_position"] = data["x_leader"] - data["x_follower"]
    data = data.sort_values(by=["case_id", "time"]).set_index("case_id")
    features = []
    labels = []
    idx = 0
    for case_id in tqdm(data.index.unique()):
        df = data.loc[case_id]
        future_idx_end = np.arange(
            40, len(df), 40
        )  # This line creates samples without overlapping, do that if the data amount is not enough or as you wish
        # future_idx_end = np.concatenate((future_idx_end, future_idx_end[1:]-15)) # make 10 timesteps overlapping, of course running time will also double
        future_idx_start = future_idx_end - 10
        history_idx_end = future_idx_start
        history_idx_start = history_idx_end - 30
        for hstart, hend, fstart, fend in zip(
            history_idx_start, history_idx_end, future_idx_start, future_idx_end
        ):
            feature = df.iloc[hstart:hend][
                ["time", "delta_velocity", "delta_position", "v_follower"]
            ].copy()
            feature["sample_id"] = idx
            label = df.iloc[fstart:fend][["time", "v_follower"]].copy()
            label["sample_id"] = idx
            features.append(feature)
            labels.append(label)
            idx += 1
    features = pd.concat(features).reset_index()
    # Standardize features
    scaler = StandardScaler()
    features[["delta_velocity", "delta_position", "v_follower"]] = scaler.fit_transform(
        features[["delta_velocity", "delta_position", "v_follower"]]
    )
    # But do not standardize labels
    labels = pd.concat(labels).reset_index()
    return features, labels


# features_HA.to_hdf(separated_data_path+'features_HA.h5', key='features') # or save in other format you are familiar with
# labels_HA.to_hdf(separated_data_path+'labels_HA.h5', key='labels')
# features_HH, labels_HH = segment_data(data_HH)
# print('number of samples in HH:', labels_HH['sample_id'].nunique())
# features_HH.to_hdf(separated_data_path+'features_HH.h5', key='features')
# labels_HH.to_hdf(separated_data_path+'labels_HH.h5', key='labels')

# %%
# Read features and labels in local
# features_HA = pd.read_hdf(separated_data_path + "features_HA.h5", key="features")
# labels_HA = pd.read_hdf(separated_data_path + "labels_HA.h5", key="labels")
# features_HH = pd.read_hdf(separated_data_path + "features_HH.h5", key="features")
# labels_HH = pd.read_hdf(separated_data_path + "labels_HH.h5", key="labels")


# %%
def data_split(features, labels):
    # Split data into training, validation, test set by idx
    # make sure the random choice of features and labels are the same!
    all_indices_HA = labels["sample_id"].unique()
    train_indices_HA = np.random.choice(
        all_indices_HA, size=int(0.7 * len(all_indices_HA)), replace=False
    )
    test_indices_HA = np.random.choice(
        np.setdiff1d(all_indices_HA, train_indices_HA),
        size=int(0.3 * len(all_indices_HA)),
        replace=False,
    )
    # val_set you can apply the previous code to val_HA that is already existing
    train_features_HA = features[features["sample_id"].isin(train_indices_HA)]
    train_labels_HA = labels[labels["sample_id"].isin(train_indices_HA)]
    test_features_HA = features[features["sample_id"].isin(test_indices_HA)]
    test_labels_HA = labels[labels["sample_id"].isin(test_indices_HA)]

    return train_features_HA, train_labels_HA, test_features_HA, test_labels_HA


# train_features_HA, train_labels_HA, test_features_HA, test_labels_HA = data_split(
#     features_HA, labels_HA
# )
# the same for HH
# ...


# %%
# Create dataloader function
class CreateDataset:
    def __init__(self, features, labels):
        self.idx_list = labels["sample_id"].unique()
        self.labels = labels.sort_values(["sample_id", "time"]).set_index("sample_id")
        self.features = features.sort_values(["sample_id", "time"]).set_index(
            "sample_id"
        )

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # idx is the index of items in the data and labels
        sample_id = self.idx_list[idx]
        history = self.features.loc[sample_id][
            ["delta_velocity", "delta_position", "v_follower"]
        ].values
        history = torch.from_numpy(history).float()
        future = self.labels.loc[sample_id]["v_follower"].values
        future = torch.from_numpy(future).float()
        return history, future


# %%
# # # Test if the dataloader works
# dataset = "HH"
# cluster_idx = 0
# train_dataloader_HA, _ = create_dataloader(dataset, cluster_idx)
# history, future = next(iter(train_dataloader_HA))
# print(f"Feature batch shape: {history.size()}")
# print(f"Labels batch shape: {future.size()}")


#
# %%
from read_data import get_train_data
from torch.utils.data import DataLoader


def prepare_dataloaders(dataset, cluster_idx, batch_size=64, shuffle_train=True):
    # 获取数据
    features = get_train_data(dataset, cluster_idx, "features")
    labels = get_train_data(dataset, cluster_idx, "labels")

    # 数据切分
    train_features, train_labels, test_features, test_labels = data_split(
        features, labels
    )

    # 创建数据集
    train_dataset = CreateDataset(train_features, train_labels)
    test_dataset = CreateDataset(test_features, test_labels)

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


# 使用示例
dataset = "HH"
cluster_idx = 0
train_dataloader, test_dataloader = prepare_dataloaders(dataset, cluster_idx)
history, future = next(iter(train_dataloader))
print(f"Feature batch shape: {history.size()}")
print(f"Labels batch shape: {future.size()}")

# %%
