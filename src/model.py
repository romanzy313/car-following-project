from __future__ import annotations
import operator
from typing import Any, List
from src.vehicle import Vehicle
import importlib


class Model:
    positions: List[float]
    velocities: List[float]
    iteration: int

    def __init__(
        self,
        id: str,
        vehicle: Vehicle,
        initial_position: float,
        inital_velocity: float,
        history_length: int = 10,
        dt: float = 0.1,
    ):
        self.iteration = 0
        self.id = id
        self.name = self.__class__.__module__[7:]
        self.dt = dt
        self.vehicle = vehicle

        # fill initial "history"
        self.positions = []
        self.velocities = []

        # backpropogate the history instead
        for i in reversed(range(0, history_length)):
            # we go backwards here
            self.positions.append(initial_position - inital_velocity * dt * i)
            self.velocities.append(inital_velocity)

    def apply_acceleration(self, acceleration: float):
        """
        This is how acceleration is applied.
        The accelerations are hard limited by min/max
        """

        next_acceleration = self.vehicle.limit_acceleration(acceleration)

        next_velocity = self.vehicle.limit_velocity(
            self.velocities[-1] + next_acceleration * self.dt
        )
        next_position = self.positions[-1] + next_velocity * self.dt

        self.velocities.append(next_velocity)
        self.positions.append(next_position)

        self.velocities.pop(0)
        self.positions.pop(0)

    def to_json(self):
        return {
            "id": self.id,
            "position": round(self.positions[-1], 2),
            "velocity": round(self.velocities[-1], 2),
        }

    def __str__(self):
        return f"[{id}] position {self.positions[-1]} velocty {self.velocities[-1]}"

    def get_deltas_with_next(self, next: Model):
        delta_positions = []

        for i in range(len(self.positions)):
            delta_positions.append(
                next.positions[i]
                - self.positions[i]
                - next.vehicle.length / 2
                - self.vehicle.length / 2
            )
        delta_velocities: Any = list(
            map(operator.sub, self.velocities, next.velocities)
        )

        if True in (t < 0 for t in delta_positions):
            print(
                f"[{self.name}] NEXT Negative delta position detected!!!",
                delta_positions,
            )

        # print("delta pos with next", delta_positions, "delta vel", delta_velocities)

        return (delta_positions, delta_velocities)

    def get_deltas_on_last(self, first: Model, road_length: float):
        last = self

        inner_deltas_pos: Any = list(map(operator.sub, last.positions, first.positions))
        delta_positions: Any = [
            *map(
                lambda x: road_length
                - x
                - self.vehicle.length / 2  # add vehicle length to it
                - first.vehicle.length / 2,
                inner_deltas_pos,
            )
        ]
        delta_velocities: Any = list(
            map(operator.sub, last.velocities, first.velocities)
        )

        if True in (t < 0 for t in delta_positions):
            print(
                f"[{self.name}] LAST Negative delta position detected!!!",
                delta_positions,
            )

        return (delta_positions, delta_velocities)

    def tick_and_get_acceleration_with_next(self, next: Model) -> float:
        (delta_positions, delta_velocities) = self.get_deltas_with_next(next)

        this_acc = self.tick(self.velocities, delta_positions, delta_velocities)

        self.iteration += 1

        # print("next acc is", this_acc, "pos", delta_positions, "vel", delta_velocities)
        return this_acc

    def tick_and_get_acceleration_on_last(
        self, first: Model, road_length: float
    ) -> float:
        (delta_positions, delta_velocities) = self.get_deltas_on_last(
            first, road_length
        )

        this_acc = self.tick(self.velocities, delta_positions, delta_velocities)
        # print("last acc is", this_acc, "pos", delta_positions, "vel", delta_velocities)

        self.iteration += 1

        return this_acc

    def check_collision_with_next(self, next: Model) -> bool:
        distance = (
            next.positions[-1]
            - next.vehicle.length / 2
            - self.positions[-1]
            - self.vehicle.length / 2
        )
        # print(self.id, "distance with next", distance)
        if distance <= 0:
            print(f"NEXT Vehicle {self.id} collided with {next.id}")
            return True

        return False

    def check_collision_on_last(self, first: Model, road_length: float) -> bool:
        last = self
        if (
            last.positions[-1]
            + last.vehicle.length / 2
            - first.positions[-1]
            + first.vehicle.length / 2
        ) >= road_length:
            # print(f"LAST Vehicle {last.id} collided with {first.id}")
            return True
        return False

    # abstract functions

    def tick(
        self,
        follower_velocities: List[float],
        delta_positions: List[float],
        delta_velocities: List[float],
    ) -> float:
        """
        Abstract function
        This is a standard model evaluation function.
        It takes in delta position and velocity to the vehicle ahead
        and returns acceleration
        """
        raise Exception("Abstract me!")

    def inject_args(self, args):
        # its okay to do nothing i guess
        pass
        # raise Exception("Abstract me!")


# https://stackoverflow.com/questions/4821104/dynamic-instantiation-from-string-name-of-a-class-in-dynamically-imported-module
def get_model_from_name(class_name: str) -> type[Model]:
    module = importlib.import_module(f"models.{class_name}")
    class_ = getattr(module, "Definition")
    return class_
