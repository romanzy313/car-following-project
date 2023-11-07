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
from torch.utils.data import DataLoader, TensorDataset
from glob import glob
import re


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
        range(epochs), position=1, leave=False, desc="training", colour="red"
    ):
        # tqdm.write("hello world")
        # time.sleep(0.2)
        # continue
        model.train()
        optimizer.zero_grad()  # Reset gradients tensors
        for i, (X_batch, y_batch) in enumerate(dataloader):
            # print(
            #     f"Batch {i} - X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}"
            # )  # Debugging line
            X_batch, y_batch = X_batch.to(device), y_batch.to(
                device
            )  # Move batch data to the device

            y_pred = model(X_batch)
            loss = (
                loss_function(y_pred, y_batch) / accumulation_steps
            )  # Normalize our loss

            loss.backward()
            if (i + 1) % accumulation_steps == 0 or i + 1 == len(dataloader):
                optimizer.step()  # Perform a single optimization step
                optimizer.zero_grad()  # Reset gradients tensors

            # Clear some memory
            del X_batch, y_batch, y_pred
            gc.collect()  # Force garbage collection
            if device == "cuda":
                torch.cuda.empty_cache()  # Clear cache if on GPU

        if epoch % 10 == 0:
            tqdm.write(
                f"[{dataset}_{cluster_idx}] Epoch: {epoch} Loss: {loss.item() * accumulation_steps:.4f}"  # type: ignore
            )  # Adjust the loss value


def run_training(
    cluster_df,
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
    if cluster_df.empty:
        raise Exception(f"Cluster {dataset}_{cluster_idx} is empty.")

    (
        X_train_tensor,
        y_train_tensor,
        X_test_tensor,
        y_test_tensor,
        scaler,
    ) = preprocess_data(cluster_df, n_steps_in, n_steps_out)

    # X_train_tensor = X_train_tensor.to(device)
    # y_train_tensor = y_train_tensor.to(device)
    # X_test_tensor = X_test_tensor.to(device)
    # y_test_tensor = y_test_tensor.to(device)
    # Create a DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # Use num_workers and pin_memory for faster data loading
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,  # or more, depending on your CPU and data
        pin_memory=True,  # helps with faster data transfer to GPU
    )

    model = Seq2Seq(
        input_size=X_train_tensor.shape[2],
        hidden_size=128,
        output_size=y_train_tensor.shape[2],
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


def train_cluster(dataset: str, cluster_idx: int, file: str):
    train_data = read_clustered_data(file)
    tqdm.write(f"[{dataset}_{cluster_idx}] Dataset size {train_data.shape}")

    # train_data = train_data.sample(frac=0.01, random_state=1)
    run_training(
        cluster_df=train_data,
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
