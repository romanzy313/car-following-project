import json

from src.scene import Scene


class SimulationRunner:
    """
    This will run the simulation until end conditions are met
    """

    time = 0
    iteration = 0
    steps = []
    outcome: str = ""

    def __init__(self, scene: Scene, dt: float, max_iterations: int) -> None:
        self.scene = scene
        self.dt = dt
        self.max_iterations = max_iterations

    def tick(self) -> bool:
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
        for i in range(0, len(models) - 1):
            this_acc = models[i].get_acceleration_with_next(models[i + 1])
            accelerations.append(this_acc)

        last_acc = models[-1].get_acceleration_on_last(
            models[0], self.scene.road_length
        )
        accelerations.append(last_acc)
        # step 3
        for i in range(0, len(models)):
            models[i].apply_acceleration(accelerations[i])

        # step 4, being lazy for now
        collided = self.check_collisions()

        # if collided:
        #     self.output.append({"end": "collision"})
        # else:

        return collided

    def check_collisions(self) -> bool:
        models = self.scene.models

        for i in range(0, len(models) - 2):
            collided = models[i].check_collision_with_next(models[i + 1])
            if collided:
                return True

        collided = models[-1].check_collision_on_last(models[0], self.scene.road_length)
        if collided:
            return True

        return False

    def run(self) -> None:
        # for every vehicle run the algorythm
        print(f"doing the run for {self.max_iterations} times")

        while True:
            collision = self.tick()
            if collision:
                # self.steps.append({"end": "collision"})
                self.outcome = "collision"
                print(f"collision at iteration {self.iteration}")
                return
            elif self.iteration == self.max_iterations:
                # self.steps.append({"end": "great success"})
                self.outcome = "ok"
                return
            else:
                self.steps.append(
                    {
                        "iteration": self.iteration,
                        "time": self.time,
                        "vehicles": list(
                            map(lambda model: model.to_json(), self.scene.models)
                        ),
                    }
                )
                self.time += self.dt
                self.iteration += 1

    def flush_to_disk(self, file: str):
        print(f"flushing simulation output to {file}")

        with open(f"{file}", "w") as fp:
            json.dump(
                {
                    "scene": self.scene.to_json(),
                    "steps": self.steps,
                    "outcome": self.outcome,
                },
                fp,
                indent=2,
            )
