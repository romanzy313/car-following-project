import random
from typing import List
from src.model import Model
from src.vehicle import Vehicle


class Definition(Model):
    spread = 0

    def inject_args(self, args):
        self.spread = args["spread"]

    def tick(
        self,
        follower_velocities: List[float],
        delta_positions: List[float],
        delta_velocities: List[float],
    ) -> float:
        """
        This sample model tries to keep the same position as the vehicle in the front
        """

        return random.uniform(-self.spread, self.spread)
