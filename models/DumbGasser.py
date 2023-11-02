from typing import List
from src.model import Model


class Definition(Model):
    gas_amount = 0

    def inject_args(self, args):
        # pass
        self.gas_amount = args["gas_amount"]
        # print("injected keep distance", self.keep_distance)

    def tick(self, delta_pos: List[float], delta_vel: List[float]) -> float:
        """
        This dumbass model just presses gas
        """

        return self.gas_amount
        # return super().tick(delta_pos, delta_vel)
