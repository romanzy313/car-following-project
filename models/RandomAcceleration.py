import random
from src.model import Model


class Definition(Model):
    spread = 0

    def inject_args(self, args):
        self.spread = args["spread"]
        # print("injected keep distance", self.keep_distance)

    def tick(self, delta_pos: float, delta_vel: float) -> float:
        """
        This sample model tries to keep the same position as the vehicle in the front
        """

        return random.uniform(0, self.spread)
        # return super().tick(delta_pos, delta_vel)
