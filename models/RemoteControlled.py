import random
from typing import Callable, List
from src.model import Model
from src.vehicle import Vehicle


class Definition(Model):
    def set_callback(self, cb: Callable[[int], float]):
        """
        First argument is current iteration

        return is what velocity must be set
        """
        self.cb = cb
        pass

    def tick(
        self,
        follower_velocities: List[float],
        delta_positions: List[float],
        delta_velocities: List[float],
    ) -> float:
        # this will smartly calculate acceleration to apply to set current velocity

        current_vel = self.velocities[-1]

        desired_vel = self.cb(self.iteration)

        needed_acc = (desired_vel - current_vel) / self.dt

        # print(f"needed velocity is {needed_acc}")

        return needed_acc
