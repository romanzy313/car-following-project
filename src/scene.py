import random
from models import RandomAcceleration
from src.vehicle import Vehicle
from src.model import Model, get_model_from_name
from models import *
from typing import Any, List

# 10 datapoints from the past are used for history (10 * 0.1 = 1 second of diving history)
# ugly hardcoded values for now
history_length = 10


class Scene:
    # speed_limit: float
    models: List[Model] = []

    # TODO collect statistics on the run
    # Like how many near misses there are

    def __init__(
        self, models: List[Model], road_length: float, dt: float, max_iterations: int
    ) -> None:
        self.models = models
        self.road_length = road_length
        self.dt = dt
        self.max_iterations = max_iterations

        # self.dt = dt
        # pass

    def to_json(self):
        def extract(model: Model):
            data = model.to_json()
            return {"id": model.id, "name": model.name}

        return {
            "road_length": self.road_length,
            "models": list(map(extract, self.models))
            # self.models[0].to_json(),
        }

    def __str__(self):
        return f"total number of models {self.models}"

    def run(self, with_steps: bool):
        time = 0.0
        iteration = 0
        collision: int | bool = False
        run = True
        steps: List[Any] = []
        while run:
            collision = self.tick()
            if collision:
                # print(f"collision at iteration {iteration}")
                run = False
            elif iteration == self.max_iterations:
                run = False
            else:
                steps.append(
                    {
                        "iteration": iteration,
                        "time": time,
                        "vehicles": list(
                            map(lambda model: model.to_json(), self.models)
                        ),
                    }
                )
                time += self.dt
                iteration += 1

        # get collision type
        collided = collision != False
        collision_follower_id = None
        collision_follower_model = None
        collision_leader_id = None
        collision_leader_model = None

        if collision:
            leader_index = collision if collision < len(self.models) - 1 else 0

            collision_follower_id = str(collision)
            collision_follower_model = self.models[collision].name
            collision_leader_id = str(leader_index)
            collision_leader_model = self.models[leader_index].name

        output = {
            "end_time": time,
            "end_iteration": iteration,
            "collided": collided,
            "collision_follower_id": collision_follower_id,
            "collision_follower_model": collision_follower_model,
            "collision_leader_id": collision_leader_id,
            "collision_leader_model": collision_leader_model,
        }

        if with_steps:
            output["steps"] = steps

        return output
        # return run results

    def tick(self):
        accelerations = []
        for i in range(0, len(self.models) - 1):
            this_acc = self.models[i].get_acceleration_with_next(self.models[i + 1])
            accelerations.append(this_acc)

        last_acc = self.models[-1].get_acceleration_on_last(
            self.models[0], self.road_length
        )
        accelerations.append(last_acc)
        # step 3
        for i in range(0, len(self.models)):
            self.models[i].apply_acceleration(accelerations[i])

        # step 4, being lazy for now
        collisionId = self.check_collisions()

        return collisionId

    def check_collisions(self) -> int | bool:
        for i in range(0, len(self.models) - 2):
            collided = self.models[i].check_collision_with_next(self.models[i + 1])
            if collided:
                return i

        collided = self.models[-1].check_collision_on_last(
            self.models[0], self.road_length
        )
        if collided:
            return len(self.models) - 1

        return False


def make_equadistent_scene(
    model_name: str,
    model_args,
    vehicle: Vehicle,
    road_length: float,
    vehicle_count: int,
    initial_velocity: float,
    fuzzy_position: float,  # by how much positions can shift
    dt: float = 0.1,
    max_iterations: int = 5000,
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
            inital_velocity=initial_velocity,
        )
        model.inject_args(model_args)

        models.append(model)

    return Scene(models, road_length, dt, max_iterations)


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

    return Scene(models, road_length, dt, max_iterations)
