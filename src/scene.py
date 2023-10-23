import random
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
        # pass

    def describe(self):
        print("total models", len(self.models))


def make_equadistent_scene(
    model_name: str,
    model_args,
    vehicle: Vehicle,
    road_length: float,
    vehicle_count: int,
    initial_velocity: float,
    fuzzy_position: float,  # by how much positions can shift
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
            vehicle=vehicle,
            initial_position=initial_position,
            inital_velocity=initial_velocity,
        )
        model.inject_args(model_args)

        models.append(model)

    return Scene(models, road_length)


import json


class SimulationRunner:
    """
    This will run the simulation until end conditions are met
    """

    time = 0
    iteration = 0
    output = []

    def __init__(self, scene: Scene) -> None:
        self.scene = scene

    def tick(self, dt: float) -> bool:
        """
        if returns true, it means that there is a collision

        1) get current deltas between the models
        taking care to make sure to account for cyclic nature of scene

        2) get all accelerations to apply from the model

        3) apply acceleration on all models

        4) calculate if there are any collisions
        if yes, halt
        if no, all all information to output
        """

        # steps 1 and 2
        models = self.scene.models
        accelerations = []
        for i in range(0, len(models) - 2):
            this = models[i]
            next = models[i + 1]
            delta_pos = next.position - this.position
            delta_vel = next.velocity - this.velocity
            this_acc = this.tick(delta_pos, delta_vel)
            accelerations.append(this_acc)
            pass

        # last one is the special case
        first = models[0]
        last = models[len(models) - 1]
        delta_pos = self.scene.road_length - (last.position - first.position)
        delta_vel = last.velocity - first.velocity
        last_acc = last.tick(delta_pos, delta_vel)
        accelerations.append(last_acc)

        # step 3
        for i in range(0, len(models) - 1):
            models[i].apply_acceleration(accelerations[i], dt)

        # step 4, being lazy for now
        collided = self.check_collisions_simple()

        # if collided:
        #     self.output.append({"end": "collision"})
        # else:

        return collided

    def check_collisions_simple(self) -> bool:
        """
        This is a simple way to check if models collide
        In this case, the length of each vehicle is not taken into account
        """

        models = self.scene.models

        for i in range(0, len(models) - 2):
            this = models[i]
            next = models[i + 1]
            if next.position - this.position <= 0:
                print(f"Vehicle {this.id} collided with {next.id}")
                return True
            pass

        # last one is the special case
        first = models[0]
        last = models[len(models) - 1]
        if self.scene.road_length - (last.position - first.position) <= 0:
            print(f"Vehicle {last.id} collided with {first.id}")
            return True

        return False

    # def apply_accelerations(self, accelerations: List[float]):
    #     for model in self.scene.models:
    #         model.app

    def run(self, dt: float, max_iterations: int):
        # for every vehicle run the algorythm
        print(f"doing the run for {max_iterations} times")

        while True:
            collision = self.tick(dt)
            if collision:
                self.output.append({"end": "collision"})
                return
            elif self.iteration == max_iterations:
                self.output.append({"end": "great success"})
                return
            else:
                self.output.append(
                    {
                        "step": {
                            "iteration": self.iteration,
                            "time": self.time,
                            "vehicles": list(
                                map(lambda model: model.json_state(), self.scene.models)
                            ),
                        }
                    }
                )
                self.time += dt
                self.iteration += 1

    def flush_to_disk(self, file: str):
        print(f"flushing simulation output to {file}")

        with open(f"{file}", "w") as fp:
            json.dump(self.output, fp, indent=2)
