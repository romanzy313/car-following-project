# %%
import numpy as np
import pandas as pd
from read_data import read_data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import zarr


def compute_delta_metrics(data):
    """
    Computes additional metrics for the dataset:
    - Delta Position: Leader's position minus Follower's position.
    - Delta Velocity: Leader's velocity minus Follower's velocity.
    - Delta Acceleration: Leader's acceleration minus Follower's acceleration.
    - Time-To-Collision (TTC): Delta Position divided by Delta Velocity.
    """
    data["delta_position"] = data["x_leader"] - data["x_follower"]
    data["delta_velocity"] = data["v_follower"] - data["v_leader"]
    data["delta_acceleration"] = data["a_follower"] - data["a_leader"]
    data["TTC"] = data["delta_position"] / data["delta_velocity"]
    data.loc[data["TTC"] < 0, "TTC"] = np.nan
    data["time_headway"] = data["delta_position"] / data["v_follower"]
    data["TTC_min"] = data["TTC"]

    # Calculate jerk for the follower vehicle
    data["jerk_follower"] = np.gradient(data["a_follower"], data["time"])
    return data


def aggregate_data_by_case(data):
    """
    Aggregates the dataset by 'case_id' to find the max and min
    of each delta metric and TTC for each case.
    Renames columns for clarity and adds case_id as a column.
    """
    aggr_data = (
        data.groupby("case_id")
        .agg(
            {
                "delta_velocity": "mean",
                "v_follower": "max",
                "delta_acceleration": "mean",
                "a_follower": "max",
                "jerk_follower": "mean",
                "time_headway": "median",
                "delta_position": "min",
                "TTC": "median",
                "TTC_min": "min",
            }
        )
        .reset_index()
    )

    return aggr_data


def adjust_ttc_sign(aggregated_data):
    """
    Ensures TTC (Time-To-Collision) is non-negative by taking the absolute value.
    """
    aggregated_data["TTC"] = aggregated_data["TTC"].abs()
    aggregated_data["TTC_min"] = aggregated_data["TTC_min"].abs()
    return aggregated_data


def convert_df(dataset: str, mode: str):
    """
    Main function that utilizes the above helper functions to preprocess the data.
    Returns a DataFrame grouped by 'case_id' with max and min values of
    delta position, delta velocity, delta acceleration, and TTC (Time-To-Collision).
    """
    data = read_data(dataset, mode)
    data = compute_delta_metrics(data)
    aggregated_data = aggregate_data_by_case(data)
    aggregated_data = adjust_ttc_sign(aggregated_data)
    return aggregated_data


# Constants for settings that may change
RANDOM_STATE = 42
PCA_COMPONENTS = 2


def preprocess_features(features):
    """Normalize the features of a dataframe."""
    scaler = StandardScaler()
    features_numeric = features.select_dtypes(include=np.number).dropna()
    normalized_data = scaler.fit_transform(features_numeric)
    return normalized_data, features_numeric


def apply_dimensionality_reduction(data, n_components=PCA_COMPONENTS):
    """Apply PCA to reduce dimensions of the data."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)


def perform_clustering(data, n_clusters):
    """Perform KMeans clustering on the data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
    labels = kmeans.fit_predict(data)
    return labels


def plot_clusters(features, labels, pca_data=None):
    """Plot the results of clustering."""
    if pca_data is not None:
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap="viridis")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title("K-Means Clustering with PCA")
    else:
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(121, projection="3d")
        ax.scatter(
            features.iloc[:, 0],
            features.iloc[:, 1],
            features.iloc[:, 2],
            c=labels,
            cmap="viridis",
            s=50,  # type: ignore
        )
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")  # type: ignore
        ax.set_title("K-Means Clustering Results")
    plt.show()


def find_optimal_clusters(data, max_clusters=5, silent=True):
    """Determine the optimal cluster count using silhouette score and elbow method."""
    inertia_list = []
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
        labels = kmeans.fit_predict(data)
        inertia_list.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))

    if silent:
        # Elbow Method Plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(2, max_clusters + 1), inertia_list, "o-")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.title("Elbow Method")

        # Silhouette Score Plot
        plt.subplot(1, 2, 2)
        plt.plot(range(2, max_clusters + 1), silhouette_scores, "o-")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Scores for Various Clusters")
        plt.tight_layout()
        plt.show()

    # Assuming the elbow is at the cluster number with the highest silhouette score
    optimal_clusters = (
        np.argmax(silhouette_scores) + 2
    )  # +2 because range starts from 2
    return optimal_clusters


def get_clustered_df(features):
    """Main function to execute the clustering analysis pipeline."""
    # Load and preprocess the data

    normalized_data, features_numeric = preprocess_features(features)

    # Optionally apply PCA
    pca_data = apply_dimensionality_reduction(normalized_data)

    # Find the optimal number of clusters
    optimal_clusters = find_optimal_clusters(pca_data)

    # Perform clustering with the optimal number of clusters
    labels = perform_clustering(pca_data, optimal_clusters)
    features_numeric["cluster"] = labels

    # Plot the results
    plot_clusters(features_numeric, labels, pca_data)

    # Compute and display the average silhouette score
    silhouette_avg = silhouette_score(pca_data, labels)
    print(f"The average silhouette_score is: {silhouette_avg}")

    return features_numeric


def train_df(dataset: str, clustered_data: pd.DataFrame, mode: str):
    """
    Returns a DataFrame with delta position, delta velocity, v_follower and cluster.
    """
    data = read_data(dataset, mode)
    data["delta_position"] = data["x_leader"] - data["x_follower"]
    data["delta_velocity"] = data["v_follower"] - data["v_leader"]

    # Merge the data with clustered_data on 'case_id' to get the 'cluster' column
    data = pd.merge(
        data, clustered_data[["case_id", "cluster"]], on="case_id", how="left"
    )
    data = data[["delta_position", "delta_velocity", "v_follower", "cluster"]]
    # print(data.head())
    return data


out_dir = "../out_cluster"


def save_AH_without_clustering():
    # save Ah as a single cluster
    data = read_data("AH", "train")
    data["delta_position"] = data["x_leader"] - data["x_follower"]
    data["delta_velocity"] = data["v_follower"] - data["v_leader"]
    runtime_data = data[["delta_velocity", "delta_position", "v_follower"]]
    file_name = f"{out_dir}/AH_0.zarr"
    print("saving AH dataset to", file_name)
    zarr.save(file_name, runtime_data)


def cluster_and_save(dataset: str):
    print("clustering dataset", dataset)
    AH_data = convert_df(dataset, "train")
    clustered_data = get_clustered_df(AH_data)
    runtime_data = train_df(dataset, clustered_data, "train")

    grouped = runtime_data.groupby("cluster")

    for cluster_value, group_df in grouped:
        # Drop the 'cluster' column
        group_df = group_df.drop(columns=["cluster"])

        # Define the filename for the CSV file
        file_name = f"{out_dir}/{dataset}_{str(cluster_value)[0]}.zarr"  # type: ignore
        print("saving cluster result to", file_name)
        # print(group_df)
        zarr.save(file_name, group_df)


# %% now cluster other datasets and save them

save_AH_without_clustering()

datasets = ["HA", "HH"]
for dataset in datasets:
    cluster_and_save(dataset)
