from typing import List
from src.Sec2SecRuntime import Seq2SeqRuntime
from src.model import Model
import torch
import pandas as pd
import numpy as np
import torch.nn as nn


class Definition(Model):
    model_type: str  # this is either A or H
    model: Seq2SeqRuntime

    def inject_args(self, args):
        # pass
        self.model_type = args["model_type"]
        model_file = args["data_file"]
        assert model_file, "data file not provided"
        self.model = Seq2SeqRuntime(model_file)
        self.name = f"ModelV1_{self.model_type}"
        # print(f"{self.name} loaded data_file {model_file}")

    def tick(
        self,
        follower_velocities: List[float],
        delta_positions: List[float],
        delta_velocities: List[float],
    ) -> float:
        runtime_data = pd.DataFrame(
            {
                "delta_position": delta_positions,
                "delta_velocity": delta_velocities,
                "v_follower": follower_velocities,
            }
        )

        prophecy = self.model.predict(runtime_data)

        # print("simulation output is", prophecy)

        result_acceleration = prophecy[0][1]

        desired_velocity = prophecy[0][2]

        # print(
        #     self.id,
        #     "current velocity",
        #     self.velocities[-1],
        #     "desired",
        #     desired_velocity,
        #     "applied acceleration",
        #     result_acceleration,
        # )

        return result_acceleration * 4
        # pre_data = pd.DataFrame(
        #     {
        #         # "l_follower": self.vehicle.length,
        #         # "l_leader": next.vehicle.length,
        #         "time": np.round(self.timestamps, 1),
        #         "x_follower": self.positions,
        #         "v_follower": self.velocities,
        #         "a_follower": self.accelerations,
        #         "x_leader": next_positions,
        #         "v_leader": next_velocities,
        #         "a_leader": next_accelerations,
        #     }
        # )

        # # print("pre_data dataframe")
        # # print(pre_data)

        # eval_df = compute_delta_metrics(pre_data)

        # # print("eval_df dataframe")
        # # print(eval_df)

        # predirected_acceleration = predict_delta_acceleration(
        #     eval_df,
        #     self.model_scalers,
        #     cluster_number=1,
        #     n_steps_in=3,
        #     delta_acceleration_index=4,
        # )

        # if np.isnan(predirected_acceleration):
        #     predirected_acceleration = 0

        # # print(f"predicted acceleration {predirected_acceleration}")

        # return predirected_acceleration
