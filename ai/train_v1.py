# %%
import numpy as np
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
        range(epochs), position=1, leave=False, desc="training", colour="red"
    ):
        model.train()
        for data in dataloader:
            """
            data is a array which shape is (batch_size*n_steps_in = 40 * features = 3)
            """
            optimizer.zero_grad()  # Reset gradients tensors

            input_seq = data[:, :30, :]  # 形状为 [256, 30, 3]
            ground_truth = data[:, -10:, :]  # 形状为 [256, 10, 3]

            # 将输入序列传递给模型
            output_seq = model(input_seq)  # 模型输出形状应该是 [256, 10, 3]

            # 计算损失，即模型输出和 ground truth 的 MSE
            loss = loss_function(output_seq, ground_truth)

            # 反向传播和优化

            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
        # 每个epoch更新loss值，或者说监控模型性能的功能
        if epoch % 10 == 0:
            tqdm.write(
                f"[{dataset}_{cluster_idx}] Epoch: {epoch} Loss: {loss.item() * accumulation_steps:.4f}"  # type: ignore
            )


def run_training(
    sequenced_dataset,
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
    tqdm.write(f"[{dataset}_{cluster_idx}] using device {device}")
    # for j in tqdm(range(10), desc="j", colour='red'):
    # time.sleep(0.5)
    # for cluster, cluster_df in clustered_dataframes.items():
    if sequenced_dataset.__len__() == 0:
        raise Exception(f"Cluster {dataset}_{cluster_idx} is empty.")

    # Create a DataLoader for batching

    train_dataset, scaler = preprocess_data(sequenced_dataset)

    # Use num_workers and pin_memory for faster data loading
    train_dataloader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,  # or more, depending on your CPU and data
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
    evaluate_model(
        dataset, cluster_idx, model, X_test_tensor, y_test_tensor, scaler, device
    )

    models_scalers = (model, scaler)

    return models_scalers


def find_all_clusters():
    all_datasets = glob(f"{cluster_dir}/*.zarr")

    result = []

    for path in all_datasets:
        match = re.match(r".*?/([AH]+)_([0-9]+)\.zarr", path)
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

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 假设我们只返回数据，没有标签
        data = self.data_list[idx]
        return torch.tensor(data, dtype=torch.float32)


def preprocess_data(df):
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(df)

    return (data_normalized, scaler)


def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(0, len(data) - n_steps_in - n_steps_out + 5):
        seq_x = data[i : i + n_steps_in]
        seq_y = data[i + n_steps_in : i + n_steps_in + n_steps_out]
        if seq_x.shape[0] == n_steps_in and seq_y.shape[0] == n_steps_out:
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)


def train_cluster(dataset: str, cluster_idx: int, file: str):
    train_data = read_clustered_data(file)
    data_points = []
    for start in range(0, len(train_data), 40):
        # 保证不越界
        end = start + 40
        if end <= len(train_data):
            data_point = train_data.iloc[start:end].values  # 将 DataFrame 转换成 NumPy 数组
            data_points.append(data_point)
    sequenced_dataset = MyCustomDataset(data_points)
    print(type(sequenced_dataset))
    print(f"This is the shape after sequenced", sequenced_dataset.__len__())

    tqdm.write(f"[{dataset}_{cluster_idx}] Dataset size {sequenced_dataset.__len__()}")

    # train_data = train_data.sample(frac=0.01, random_state=1)
    run_training(
        sequenced_dataset=sequenced_dataset,  # type: ignore
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
epochs = 100
lr = 0.01
device = "auto"
num_workers = multiprocessing.cpu_count()

# set the dataset and mode
if __name__ == "__main__":
    datas = find_all_clusters()
    os.makedirs(brain_dir, exist_ok=True)
    for v in tqdm(datas, position=0, leave=False, desc="per cluster", colour="green"):
        dataset = v["dataset"]
        cluster_idx = v["cluster"]
        file = v["file"]
        train_cluster(dataset, cluster_idx, file)
