from typing import List
from ai.Sec2SecRuntime import Seq2SeqRuntime
from src.model import Model
import pandas as pd
import numpy as np


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

        # desired_velocity = prophecy[0][2]

        # print(
        #     self.id,
        #     "delta_position",
        #     delta_positions,
        #     "applied acceleration",
        #     result_acceleration,
        # )

        if result_acceleration < 0:
            print(self.id, "is breaking!", round(result_acceleration, 2))

        # boosted was removed
        return result_acceleration
