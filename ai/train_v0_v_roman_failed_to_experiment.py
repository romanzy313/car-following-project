# %%
import numpy as np
from Sec2SecRuntime import Seq2Seq
from read_data import get_scaler, get_train_data
import multiprocessing
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from tqdm import tqdm
import math
from torch.utils.data import DataLoader, TensorDataset
from glob import glob
import re


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


def preprocess_data(df, n_steps_in=30, n_steps_out=10, test_size=0.2):
    # 只保留需要的列
    df = df[["delta_position", "delta_velocity", "v_follower"]]
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(df)
    X, y = create_sequences(data_normalized, n_steps_in, n_steps_out)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        scaler,
    )


def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(0, len(data) - n_steps_in - n_steps_out + 5):
        seq_x = data[i : i + n_steps_in]
        seq_y = data[i + n_steps_in : i + n_steps_in + n_steps_out]
        if seq_x.shape[0] == n_steps_in and seq_y.shape[0] == n_steps_out:
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)


def evaluate_model(
    dataset,
    cluster_idx,
    model,
    X_test_tensor,
    y_test_tensor,
    scaler,
    device,
    batch_size=1024,
):
    # Move the model to the specified device
    model = model.to(device)

    # Create a DataLoader for the test data
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    y_pred_list = []
    y_test_list = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Move the tensors to the same device as the model
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Predict
            y_pred = model(X_batch)
            # Move the predictions back to CPU
            y_pred_list.append(y_pred.cpu())
            y_test_list.append(y_batch.cpu())

    # Concatenate all batches
    y_pred_numpy = torch.cat(y_pred_list).numpy()
    y_test_numpy = torch.cat(y_test_list).numpy()

    # Inverse transform the predictions and true values
    y_pred_original = scaler.inverse_transform(
        y_pred_numpy.reshape(-1, y_pred_numpy.shape[-1])
    ).reshape(y_pred_numpy.shape)
    y_test_original = scaler.inverse_transform(
        y_test_numpy.reshape(-1, y_test_numpy.shape[-1])
    ).reshape(y_test_numpy.shape)

    # Calculate metrics
    mse = mean_squared_error(y_test_original[:, 0, :], y_pred_original[:, 0, :])
    rmse = math.sqrt(mse)  # type: ignore
    mae = mean_absolute_error(y_test_original[:, 0, :], y_pred_original[:, 0, :])

    tqdm.write(
        f"[{dataset}_{cluster_idx}] Evaluation. MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}"
    )


def train_model(
    model,
    dataloader,
    epochs,
    optimizer,
    loss_function,
    device,
    accumulation_steps=4,
):
    model.to(device)  # Ensure model is on the correct device

    for epoch in tqdm(
        range(epochs), position=1, leave=False, desc="Training", colour="red"
    ):
        model.train()
        for history, future in dataloader:
            # Since the data loader already separates history and future, no slicing is needed
            input_seq = history.to(device)
            ground_truth = future.to(device)

            output_seq = model(input_seq)

            loss = loss_function(output_seq, ground_truth)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch % 10 == 0:
            tqdm.write(
                f"[{dataset}_{cluster_idx}] Epoch: {epoch} Loss: {loss.item() * accumulation_steps:.4f}"  # type: ignore
            )  # Adjust the loss value


def run_training(
    # cluster_df,
    dataset,
    cluster_idx,
    n_steps_in,
    n_steps_out,
    epochs,
    lr,
    device,
    num_workers,
):
    """
    Autodevice will try to use cuda if possible, otherwise uses what is specified
    """
    # print("device specified", device)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    tqdm.write(
        f"[{dataset}_{cluster_idx}] using device {device} and {num_workers} workers"
    )
    # for j in tqdm(range(10), desc="j", colour='red'):
    # time.sleep(0.5)
    # for cluster, cluster_df in clustered_dataframes.items():

    # scaler not needed
    scaler = get_scaler(dataset, cluster_idx)
    dataset = CreateDataset(
        get_train_data(dataset, cluster_idx, "features"),
        get_train_data(dataset, cluster_idx, "labels"),
    )
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)  # type: ignore

    model = Seq2Seq(
        input_size=3,
        hidden_size=128,
        output_size=3,
        n_steps_out=10,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    model.to(device)  # Move model to the device (GPU or CPU)

    train_model(
        model,
        train_dataloader,  # Pass the DataLoader instead of tensors directly
        epochs,
        optimizer,
        loss_function,
        device,  # Pass the device to the training function
    )

    file_location = f"{brain_dir}/{dataset}_{cluster_idx}.pth"
    # Save the model and scaler for this cluster
    torch.save(
        {"model_state_dict": model.state_dict(), "scaler": scaler},
        file_location,
    )
    # evaluate_model(
    #     dataset, cluster_idx, model, X_test_tensor, y_test_tensor, scaler, device
    # )

    models_scalers = (model, scaler)

    return models_scalers


def find_all_clusters():
    all_datasets = glob(f"{cluster_dir}/*.zarr")

    result = []

    for path in all_datasets:
        match = re.match(r".*?/([AH|HA|HH]+)_([0-9]+)\.zarr", path)
        if match:
            dataset_name, cluster = match.groups()
            cluster = int(cluster)
            result.append({"dataset": dataset_name, "cluster": cluster, "file": path})

    return result
    # extract names from it too
    # for i in tqdm(datasets, position=0, leave=False, desc="i", colour="green"):


def train_cluster(dataset: str, cluster_idx: int, file: str):
    # train_data = read_clustered_data(file)
    # tqdm.write(f"[{dataset}_{cluster_idx}] Dataset size {train_data.shape}")

    # train_data = train_data.sample(frac=0.01, random_state=1)
    run_training(
        # cluster_df=train_data,
        dataset=dataset,
        cluster_idx=cluster_idx,
        n_steps_in=n_steps_in,
        n_steps_out=n_steps_out,
        epochs=epochs,
        lr=lr,
        device=device,
        num_workers=num_workers,
    )


# Global settings are here
cluster_dir = "../out_cluster"
brain_dir = "../out_brain"
n_steps_in = 30
n_steps_out = 10
epochs = 30
lr = 0.01
device = "auto"
num_workers = multiprocessing.cpu_count()

# set the dataset and mode
if __name__ == "__main__":
    # datas = find_all_clusters()
    datas = [{"dataset": "AH", "cluster": 0, "file": "../out_cluster/AH_0.zarr"}]
    print("running clustering on following datasets:", datas)
    os.makedirs(brain_dir, exist_ok=True)
    for v in tqdm(datas, position=0, leave=False, desc=" cluster", colour="green"):
        dataset = v["dataset"]
        cluster_idx = v["cluster"]
        file = v["file"]
        train_cluster(dataset, cluster_idx, file)
