from typing import List
from src.model import Model
import torch
import pandas as pd
import numpy as np
import torch.nn as nn


# copied from LSTM
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


def predict_delta_acceleration(
    eval_df, models_scalers, cluster_number=1, n_steps_in=3, delta_acceleration_index=2
):
    """
    Predicts the delta acceleration of a car using an LSTM model trained on car-following data.

    Parameters:
    eval_df (pandas.DataFrame): The input data to predict on.
    models_scalers (dict): A dictionary containing the trained models and scalers for each cluster.
    cluster_number (int): The cluster number to use for prediction.
    n_steps_in (int): The number of time steps to use as input for the LSTM model.
    delta_acceleration_index (int): The index of the delta acceleration column in the output.

    Returns:
    float: The predicted delta acceleration.
    """

    # Load the scaler for the cluster
    # scaler = models_scalers[cluster_number]["scaler"]

    # # Prepare the input data for prediction
    # X_new_prepared = preprocess_new_data(eval_df.values, scaler, n_steps_in)
    # X_new_tensor = torch.tensor(X_new_prepared, dtype=torch.float32)

    # # Load the model for the cluster
    # model = models_scalers[cluster_number]["model"]

    # # Predict using the model
    # model.eval()
    # with torch.no_grad():
    #     y_new_pred_tensor = model(X_new_tensor)
    #     y_new_pred = y_new_pred_tensor.numpy()

    # # Inverse transform the predictions to the original scale
    # y_new_pred_original = scaler.inverse_transform(y_new_pred)

    # # Extract the denormalized delta_acceleration values
    # delta_acceleration_pred_original = y_new_pred_original[:, delta_acceleration_index]

    # # Return the predicted delta acceleration
    # return delta_acceleration_pred_original

    # for now just return 0
    return 0


class Definition(Model):
    model_type: str  # this is either A or H

    def inject_args(self, args):
        # pass
        self.model_type = args["model_type"]
        model_file = args["data_file"]

        self.model_scalers = torch.load(model_file)
        self.name = f"ModelV1_{self.model_type}"
        print(f"{self.name} loaded data_file {model_file}")

    # this really needs
    # p_follower
    # v_follower
    # a_follower
    # delta_position
    # delta_velocity
    # delta_acceleration
    # jerk_follower data['jerk_follower'] = np.gradient(data['a_follower'], data['time'])
    # time_headway data['time_headway'] = data['delta_position'] / data['v_follower']
    # TTC     data["TTC"] = data["delta_position"] / data["delta_velocity"]
    # TTC_min data['TTC_min'] = data['TTC']???
    def tick(
        self,
        # next: Model,
        next_positions: List[float],
        next_velocities: List[float],
        next_accelerations: List[float],  # this is a frame behind but its okay?
    ) -> float:
        pre_data = pd.DataFrame(
            {
                # "l_follower": self.vehicle.length,
                # "l_leader": next.vehicle.length,
                "p_follower": self.positions,
                "v_follower": self.velocities,
                "a_follower": self.accelerations,
                "x_leader": next_positions,
                "v_leader": next_velocities,
                "a_leader": next_accelerations,
            }
        )

        eval_df = compute_delta_metrics(pre_data)

        predirected_acceleration = predict_delta_acceleration(
            eval_df, self.model_scalers
        )

        print(f"predicted acceleration {predirected_acceleration}")

        return predirected_acceleration
