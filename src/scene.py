from src.vehicle import Vehicle
from src.model import Model
from typing import List


class Scene:
    # speed_limit: float

    models: List[Model] = []

    def __init__(self, models: List[Model]) -> None:
        self.models = models
        # pass

    def describe(self):
        print("total models", len(self.models))


def make_equadistent_scene(
    vehicle: Vehicle, road_length: float, vehicle_count: int, initial_velocity: float
):
    models: List[Model] = []
    segment_length = road_length / vehicle_count

    for i in range(0, vehicle_count):
        initial_position = i * segment_length
        model = Model(
            id=str(i),
            length=vehicle.length,
            max_acceleration=vehicle.max_acceleration,
            max_deceleration=vehicle.max_deceleration,
            initial_position=initial_position,
            inital_velocity=initial_velocity,
        )

        models.append(model)

    return Scene(models)


import json


class SimulationRunner:
    """
    This will run the simulation until end conditions are met
    """

    output = []

    def __init__(self, scene: Scene, dt: float, max_iterations: int) -> None:
        self.scene = scene
        self.dt = dt
        self.time = 0
        self.iterations = max_iterations
        pass

    def run(self):
        # for every vehicle run the algorythm
        print("doing the run")
        self.output.append({"hello": "world"})
        # first lets calculate the distances between each

        pass

    def flush_to_disk(self, file: str):
        print(f"flushing things {file}")

        with open(f"{file}", "w") as fp:
            json.dump(self.output, fp)
