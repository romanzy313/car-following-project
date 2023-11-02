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
        next_positions: List[float],
        next_velocities: List[float],
        next_accelerations: List[float],
    ) -> float:
        """
        This sample model tries to keep the same position as the vehicle in the front
        """

        return random.uniform(0, self.spread)
