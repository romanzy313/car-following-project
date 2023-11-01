import json
from typing import Any

from src.scene import Scene


class SimulationRunner:
    """
    This will run the simulation until a collision is found
    """

    results: Any

    def __init__(self, scene: Scene, dt: float, max_iterations: int) -> None:
        self.scene = scene
        self.dt = dt
        self.max_iterations = max_iterations

    def run(self) -> None:
        # for every vehicle run the algorythm
        print(f"doing the run for maximum {self.max_iterations} iterations")

        self.results = self.scene.run(with_steps=True)

    def get_results(self):
        return {
            "scene": self.scene.to_json(),
            **self.results,
        }

    def flush_to_disk(self, file: str):
        print(f"writing simulation output to {file}")

        with open(f"{file}", "w") as fp:
            json.dump(
                self.get_results(),
                fp,
                indent=2,
            )
