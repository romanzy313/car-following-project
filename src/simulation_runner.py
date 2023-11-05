import json
from typing import Any

from src.scene import Scene


class SimulationRunner:
    """
    This will run the simulation until a collision is found
    """

    results: Any

    def __init__(
        self,
        scene: Scene,
        with_steps: bool,
        with_statistics: bool,
        display_progress: bool,
        dt: float = 0.1,
    ) -> None:
        self.scene = scene
        self.dt = dt
        self.with_steps = with_steps
        self.with_statistics = with_statistics
        self.display_progress = display_progress

    def run(self) -> bool:
        # for every vehicle run the algorythm

        self.results = self.scene.run(
            with_steps=self.with_steps,
            with_statistics=self.with_statistics,
            display_progress=self.display_progress,
        )

        success = self.results["collided"] is None
        return success

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
