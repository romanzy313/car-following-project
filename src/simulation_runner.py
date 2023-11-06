import json
from typing import Any

from src.scene import Scene
from pathlib import Path


class SimulationRunner:
    """
    This will run the simulation until a collision is found
    """

    results: Any
    wasRan: bool

    def __init__(
        self,
        scene: Scene,
        with_steps: bool = True,
        with_statistics: bool = False,
        display_progress: bool = False,
        dt: float = 0.1,
    ) -> None:
        self.scene = scene
        self.dt = dt
        self.with_steps = with_steps
        self.with_statistics = with_statistics
        self.display_progress = display_progress

        self.wasRan = False

    def run(self) -> bool:
        # for every vehicle run the algorythm

        self.results = self.scene.run(
            with_steps=self.with_steps,
            with_statistics=self.with_statistics,
            display_progress=self.display_progress,
        )
        self.wasRan = True

        success = self.results["collided"] is False
        return success

    def did_collide(self) -> bool:
        assert self.wasRan == True, "run simulation first"
        print("collided is", self.results["collided"])
        return self.results["collided"] is True

    def get_position_with_first(self):
        assert self.wasRan == True, "run simulation first"
        assert len(self.scene.models) >= 2, "need to have atleast 2 models defined"

        positions = self.scene.get_model_positions()
        delta = positions[1] - positions[0]

        return delta

    def get_results(self):
        return {
            "scene": self.scene.to_json(),
            **self.results,
        }

    def get_result_value(self, name: str) -> float:
        return self.results[name]

    def flush_to_disk(self, mark_failed=False):
        Path("./test_results").mkdir(parents=False, exist_ok=True)
        failed_extra = "_failed" if mark_failed and self.did_collide() else ""
        self.flush_to_disk_to_file(
            f"./test_results/{self.scene.name}{failed_extra}.json"
        )

    def flush_to_disk_to_file(self, file: str):
        print(f"writing simulation output to {file}")

        with open(f"{file}", "w") as fp:
            json.dump(
                self.get_results(),
                fp,
                indent=2,
            )
