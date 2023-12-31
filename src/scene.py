import math
import random

import numpy as np
from src.vehicle import Vehicle
from src.model import Model, get_model_from_name
from typing import Any, List
from statistics import mean, median
from tqdm import tqdm

# 10 datapoints from the past are used for history (10 * 0.1 = 1 second of diving history)
# ugly hardcoded values for now
history_length = 10


class SceneStat:
    total: float
    count: int
    max: float
    min: float

    def __init__(self) -> None:
        self.total = 0
        self.count = 0
        self.max = -math.inf
        self.min = math.inf
        pass

    def append(self, val: float):
        if math.isfinite(val):
            self.total += val
            self.count += 1
            if val > self.max:
                self.max = val
            elif val < self.min:
                self.min = val

    def collect(self, name: str):
        if self.count == 0:
            return {
                f"{name}_mean": None,
                f"{name}_max": None,
                f"{name}_min": None,
            }

        return {
            f"{name}_mean": round(self.total / self.count, 2),
            f"{name}_max": round(self.max, 2),
            f"{name}_min": round(self.min, 2),
        }


class Scene:
    # speed_limit: float
    models: List[Model]

    # TODO collect statistics on the run
    # Like how many near misses there are

    def __init__(
        self,
        models: List[Model],
        road_length: float,
        max_iterations: int,
        name: str = "unnamed",
        dt: float = 0.1,
    ) -> None:
        self.name = name
        self.models = models
        self.road_length = road_length
        self.dt = dt
        self.max_iterations = max_iterations

        self.stat_velocity = SceneStat()
        self.stat_ttc = SceneStat()
        self.stat_delta_pos = SceneStat()

        # self.dt = dt
        # pass

    def to_json(self):
        def extract(model: Model):
            return {
                "id": model.id,
                "name": model.name,
                "display": model.vehicle.display,
            }

        return {
            "road_length": self.road_length,
            "models": list(map(extract, self.models))
            # self.models[0].to_json(),
        }

    def __str__(self):
        return f"total number of models {self.models}"

    def run(
        self,
        with_steps: bool = True,
        with_statistics: bool = True,
        display_progress: bool = False,
    ):
        time = 0.0
        collision: int | None = None
        steps: List[Any] = []
        iteration = 0

        for iteration in tqdm(range(self.max_iterations), disable=not display_progress):
            if with_statistics:
                self.sample_metrics()
            if with_steps:
                steps.append(
                    {
                        "iteration": iteration,
                        "time": round(time, 1),
                        "vehicles": list(
                            map(lambda model: model.to_json(), self.models)
                        ),
                        # maybe lets append the statistics here?
                    }
                )
            time += self.dt
            collision = self.tick()
            if collision is not None:
                break

        collided = collision is not None
        collision_follower_id = None
        collision_follower_model = None
        collision_leader_id = None
        collision_leader_model = None

        if collision is not None:
            leader_index = collision if collision < len(self.models) - 1 else 0

            collision_follower_id = str(collision)
            collision_follower_model = self.models[collision].name
            collision_leader_id = str(leader_index)
            collision_leader_model = self.models[leader_index].name

        output = {
            "progress": (iteration + 1) / self.max_iterations,
            "end_time": round(time, 1),
            "end_iteration": iteration + 1,
            "collided": collided,
            "collision_follower_id": collision_follower_id,
            "collision_follower_model": collision_follower_model,
            "collision_leader_id": collision_leader_id,
            "collision_leader_model": collision_leader_model,
        }

        if with_steps:
            output["steps"] = steps
        if with_statistics:
            output.update(self.collect_metrics())

        return output
        # return run results

    def tick(self):
        accelerations = []
        for i in range(0, len(self.models) - 1):
            this_acc = self.models[i].tick_and_get_acceleration_with_next(
                self.models[i + 1]
            )
            accelerations.append(this_acc)

        last_acc = self.models[-1].tick_and_get_acceleration_on_last(
            self.models[0], self.road_length
        )
        accelerations.append(last_acc)

        # print("ticking accelerations", accelerations)

        # step 3
        for i in range(0, len(self.models)):
            self.models[i].apply_acceleration(accelerations[i])

        collisionId = self.check_collisions()

        return collisionId

    def check_collisions(self) -> int | None:
        for i in range(0, len(self.models) - 1):
            collided = self.models[i].check_collision_with_next(self.models[i + 1])
            if collided:
                return i

        collided = self.models[-1].check_collision_on_last(
            self.models[0], self.road_length
        )
        if collided:
            return len(self.models) - 1

        return None

    def sample_metrics(self):
        # metrics such as average velocity and average time to collision
        # just add all of them
        def add_ttc(delta_positions, delta_velocities):
            if delta_velocities[-1] != 0:
                ttc = delta_positions[-1] / delta_velocities[-1]
                # extra checks to avoid blowing it out
                # Yiru said this is ok
                if ttc > 0 and ttc < 10:
                    self.stat_ttc.append(ttc)

        for model in self.models:
            self.stat_velocity.append(model.velocities[-1])

        # compute ttc here...

        for i in range(len(self.models) - 1):
            (delta_positions, delta_velocities) = self.models[i].get_deltas_with_next(
                self.models[i + 1]
            )
            add_ttc(delta_positions, delta_velocities)
            self.stat_delta_pos.append(delta_positions[-1])
            # print("deltas", delta_positions, delta_velocities)

        (delta_positions, delta_velocities) = self.models[-1].get_deltas_on_last(
            self.models[0], self.road_length
        )
        add_ttc(delta_positions, delta_velocities)
        self.stat_delta_pos.append(delta_positions[-1])

    def collect_metrics(self):
        return {
            **self.stat_velocity.collect("velocity"),
            **self.stat_ttc.collect("ttc"),
            **self.stat_delta_pos.collect(
                "delta_pos",
            ),
        }

    def get_model_positions(self):
        return list(map(lambda x: x.positions[-1], self.models))


def make_equadistent_scene(
    scene_name: str,
    model_name: str,
    model_args,
    vehicle: Vehicle,
    road_length: float,
    vehicle_count: int,
    initial_velocity: float,
    fuzzy_position: float,  # by how much positions can shift
    max_iterations: int,
    dt: float = 0.1,
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
            history_length=history_length,  # hardcoded for now
            vehicle=vehicle,
            initial_position=initial_position,
            inital_velocity=initial_velocity + random.uniform(-0.1, 0.1),
        )
        model.inject_args(model_args)
        models.append(model)

    return Scene(models, road_length, max_iterations, scene_name)


def make_a_b_scene(
    model_a_name: str,
    model_b_name: str,
    a_percentage: float,
    vehicle_count: int,
    vehicle: Vehicle,
    road_length: float,
    initial_velocity: float,
    dt: float = 0.1,
    random_seed: int = 0,
    max_iterations: int = 5000,
    model_a_args={},
    model_b_args={},
):
    model_a_class = get_model_from_name(model_a_name)
    model_b_class = get_model_from_name(model_b_name)

    models: List[Model] = []
    segment_length = road_length / vehicle_count

    if random_seed != 0:
        random.seed(random_seed)

    model_choices = random.choices(
        ["a", "b"], weights=[a_percentage, 1 - a_percentage], k=vehicle_count
    )

    for i in range(0, vehicle_count):
        initial_position = i * segment_length

        choice = model_choices[i]

        # randomly choose a model to make

        model_class = model_a_class if choice == "a" else model_b_class

        model = model_class(
            id=f"{i}{choice}",
            dt=dt,
            history_length=history_length,  # hardcoded for now
            vehicle=vehicle,
            initial_position=initial_position,
            inital_velocity=initial_velocity,
        )
        model_args = model_a_args if choice == "a" else model_b_args
        model.inject_args(model_args)

        models.append(model)

    return Scene(models, road_length, max_iterations)
