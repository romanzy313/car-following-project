# %%
import numpy as np
from pandas import DataFrame
from Sec2SecRuntime import Seq2Seq
from read_data import read_clustered_data
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
from torch.utils.data import DataLoader, TensorDataset, Dataset
from glob import glob
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    scaler,
    epochs,
    optimizer,
    loss_function,
    device,
    accumulation_steps=4,
):
    model.to(device)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

    patience = 20
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in tqdm(
        range(epochs), position=1, leave=False, desc="Training", colour="red"
    ):
        model.train()
        for history, future in train_dataloader:
            optimizer.zero_grad()

            # Since the data loader already separates history and future, no slicing is needed
            input_seq = history.to(device)
            ground_truth = future.to(device)

            output_seq = model(input_seq)

            loss = loss_function(output_seq, ground_truth)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        val_loss = 0
        mse_errors = []

        with torch.no_grad():
            for history, future in val_dataloader:
                input_seq = history.to(device)
                ground_truth = future.to(device)

                output_seq = model(input_seq)

                val_loss += loss_function(output_seq, ground_truth).item()

                # The following normalization and MSE calculation can be refactored into a function
                # to reduce redundancy between training and validation loops.
                mse_error = calculate_denormalized_mse(output_seq, ground_truth, scaler)

                # Calculate MSE error and append to list
                mse_errors.append(mse_error)

        # Compute average losses
        val_loss /= len(val_dataloader)
        avg_mse_error = sum(mse_errors) / len(mse_errors)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break

        # Update learning rate
        scheduler.step(val_loss)

        # Print epoch stats
        print(
            f"[{dataset}_{cluster_idx}] Epoch: {epoch} Loss: {loss.item():.4f} Val Loss: {val_loss:.4f} Avg MSE: {avg_mse_error:.4f}"
        )


def run_training(
    train_seq,
    val_seq,
    scaler,
    dataset,
    cluster_idx,
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
    tqdm.write(f"[{dataset}_{cluster_idx}] using device {device}")
    # for j in tqdm(range(10), desc="j", colour='red'):
    # time.sleep(0.5)
    # for cluster, cluster_df in clustered_dataframes.items():
    if train_seq.__len__() == 0:
        raise Exception(f"Cluster {dataset}_{cluster_idx} is empty.")

    # Create a DataLoader for batching

    # Use num_workers and pin_memory for faster data loading
    train_dataloader = DataLoader(
        dataset=train_seq,  # type: ignore
        batch_size=128,
        shuffle=False,
        num_workers=8,  # or more, depending on your CPU and data
        pin_memory=True,  # helps with faster data transfer to GPU
        persistent_workers=True,
    )

    eval_dataloader = DataLoader(
        dataset=val_seq,  # type: ignore
        batch_size=128,
        shuffle=False,
        num_workers=8,  # or more, depending on your CPU and data
        pin_memory=True,  # helps with faster data transfer to GPU
    )

    model = Seq2Seq(
        input_size=3,
        hidden_size=128,
        output_size=3,
        n_steps_out=n_steps_out,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    model.to(device)  # Move model to the device (GPU or CPU)

    train_model(
        model,
        train_dataloader,  # Pass the DataLoader instead of tensors directly
        eval_dataloader,
        scaler,
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

    # evaluate_model(dataset, cluster_idx, model, scaler, device, eval_dataloader)
    # type: ignore
    models_scalers = (model, scaler)

    return models_scalers


def calculate_denormalized_mse(output, target, scaler):
    """
    Calculate the Mean Squared Error (MSE) between the model's outputs and the ground truth
    after denormalizing the data.

    Parameters:
    - output: the output tensor from the model
    - target: the ground truth tensor
    - scaler: the scaler instance used for normalizing/denormalizing the data

    Returns:
    - mse: the calculated mean squared error after denormalization
    """
    # Move tensors to CPU and convert to numpy for sklearn compatibility
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()

    # Denormalize the data
    output_denorm = scaler.inverse_transform(output_np.reshape(-1, output_np.shape[-1]))
    target_denorm = scaler.inverse_transform(target_np.reshape(-1, target_np.shape[-1]))

    # Calculate MSE
    mse = mean_squared_error(target_denorm, output_denorm)

    return mse


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


class MyCustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        # 假设所有数据点形状相同，取第一个数据点的形状
        self._shape = torch.tensor(data_list[0], dtype=torch.float32).shape

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 假设我们只返回数据，没有标签
        data = self.data_list[idx]
        return torch.tensor(data, dtype=torch.float32)

    @property
    def shape(self):
        # 返回数据集中每个项的形状以及总项数
        return (self.__len__(),) + self._shape


def preprocess_data(df):
    scaler = StandardScaler()

    data_normalized = scaler.fit_transform(df)
    train_df, val_df = train_test_split(data_normalized, test_size=0.2, random_state=42)
    train_df = DataFrame(train_df)
    val_df = DataFrame(val_df)
    return (train_df, val_df, scaler)


def create_sequences(data):
    data_points = []
    for start in range(0, len(data), 40):
        # 保证不越界
        end = start + 40
        if end <= len(data):
            data_point = data.iloc[start:end].values  # type: ignore # 将 DataFrame 转换成 NumPy 数组
            data_points.append(data_point)
    return data_points


def train_cluster(dataset: str, cluster_idx: int, file: str):
    train_data = read_clustered_data(file)
    train_df, val_df, scaler = preprocess_data(train_data)

    train_seq = create_sequences(train_df)
    val_seq = create_sequences(val_df)

    train_seq = MyCustomDataset(train_seq)
    val_seq = MyCustomDataset(val_seq)

    # sequenced_dataset = MyCustomDataset(data_points)
    print(type(train_seq))
    print(f"This is the shape after sequenced", train_seq.__len__())

    tqdm.write(f"[{dataset}_{cluster_idx}] Dataset size {train_seq.__len__()}")

    # train_data = train_data.sample(frac=0.01, random_state=1)
    run_training(
        train_seq=train_seq,
        val_seq=val_seq,
        scaler=scaler,  # type: ignore
        dataset=dataset,
        cluster_idx=cluster_idx,
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
epochs = 100
lr = 0.01

device = "auto"
num_workers = multiprocessing.cpu_count() / 2

# set the dataset and mode
if __name__ == "__main__":
    # datas = find_all_clusters()
    datas = [{"dataset": "HA", "cluster": 0, "file": "../out_cluster/AH_0.zarr"}]
    os.makedirs(brain_dir, exist_ok=True)
    for v in tqdm(datas, position=0, leave=False, desc="per cluster", colour="green"):
        dataset = v["dataset"]
        cluster_idx = v["cluster"]
        file = v["file"]
        train_cluster(dataset, cluster_idx, file)
