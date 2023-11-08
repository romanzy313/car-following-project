# %%
import numpy as np
from pandas import DataFrame
from Sec2SecRuntime import Seq2Seq
import multiprocessing
import os
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
from Split_dataloader import prepare_dataloaders


def train_model(
    model,
    train_dataloader,
    val_dataloader,
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

            loss = loss_function(output_seq[:, :, 0], ground_truth)
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

                val_loss += loss_function(output_seq[:, :, 0], ground_truth).item()

                # The following normalization and MSE calculation can be refactored into a function
                # to reduce redundancy between training and validation loops.
                mse_error = torch.nn.functional.mse_loss(
                    output_seq[:, :, 0], ground_truth
                )

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
            f"[{dataset}_{cluster_idx}] Epoch: {epoch} Loss: {loss.item():.4f} Val Loss: {val_loss:.4f} Avg MSE: {avg_mse_error:.4f}"  # type: ignore
        )


def run_training(
    train_dataloader,
    eval_dataloader,
    dataset,
    cluster_idx,
    n_steps_out,
    epochs,
    lr,
    device,
):
    """
    Autodevice will try to use cuda if possible, otherwise uses wtH is specified
    """
    # print("device specified", device)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    tqdm.write(f"[{dataset}_{cluster_idx}] using device {device}")
    # for j in tqdm(range(10), desc="j", colour='red'):
    # time.sleep(0.5)
    # for cluster, cluster_df in clustered_dataframes.items():
    if train_dataloader.__len__() == 0:
        raise Exception(f"Cluster {dataset}_{cluster_idx} is empty.")

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
        epochs,
        optimizer,
        loss_function,
        device,  # Pass the device to the training function
    )

    file_location = f"{brain_dir}/{dataset}_{cluster_idx}.pth"
    # Save the model for this cluster
    torch.save(
        {"model_state_dict": model.state_dict()},
        file_location,
    )


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


def train_cluster(dataset: str, cluster_idx: int, train_dataloader, eval_dataloader):
    run_training(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        dataset=dataset,
        cluster_idx=cluster_idx,
        n_steps_out=n_steps_out,
        epochs=epochs,
        lr=lr,
        device=device,
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
    dataset = "HH"
    cluster_idx = 0
    # datas = [{"dataset": "AH", "cluster": 0, "file": "../out_cluster/AH_0.zarr"}]

    train_dataloader, eval_dataloader = prepare_dataloaders(dataset, cluster_idx)
    train_cluster(dataset, cluster_idx, train_dataloader, eval_dataloader)
    # os.makedirs(brain_dir, exist_ok=True)
    # for v in tqdm(datas, position=0, leave=False, desc="per cluster", colour="green"):
    #     dataset = v["dataset"]
    #     cluster_idx = v["cluster"]
    #     file = v["file"]
    #     train_cluster(dataset, cluster_idx, train_dataloader, eval_dataloader)
