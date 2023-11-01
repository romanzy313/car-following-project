import random
from models import RandomAcceleration
from src.vehicle import Vehicle
from src.model import Model, get_model_from_name
from models import *
from typing import List


class Scene:
    # speed_limit: float
    models: List[Model] = []
    road_length: float

    def __init__(self, models: List[Model], road_length: float) -> None:
        self.models = models
        self.road_length = road_length
        # self.dt = dt
        # pass

    def to_json(self):
        return {
            "road_length": self.road_length,
            "models": list(map(lambda model: model.to_json()["id"], self.models))
            # self.models[0].to_json(),
        }

    def print_state(self):
        print(f"total number of models {self.models}")


def make_equadistent_scene(
    model_name: str,
    model_args,
    vehicle: Vehicle,
    road_length: float,
    vehicle_count: int,
    initial_velocity: float,
    fuzzy_position: float,  # by how much positions can shift
    dt: float,
):
    model_class = get_model_from_name(model_name)
    models: List[Model] = []
    segment_length = road_length / vehicle_count

    for i in range(0, vehicle_count):
        initial_position = i * segment_length

        # randomize positions a bit
        initial_position += random.uniform(-fuzzy_position, fuzzy_position)
        model = model_class(
            id=str(i),
            dt=dt,
            history_length=4,  # hardcoded for now
            vehicle=vehicle,
            initial_position=initial_position,
            inital_velocity=initial_velocity,
        )
        model.inject_args(model_args)

        models.append(model)

    return Scene(models, road_length)
