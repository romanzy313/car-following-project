from typing import Any, List
from multiprocessing import Process, Array
from src.scene import Scene
import glob
from src.simulation_runner import SimulationRunner
import os


class TestRunner:
    runners: List[SimulationRunner]

    def __init__(self, quick_fail: bool) -> None:
        self.quick_fail = quick_fail
        self.runners = []

        pass

    def add_scene(
        self,
        scene: Scene,
    ):
        self.runners.append(SimulationRunner(scene, True, False, True))

    def run_scene(self, runner: SimulationRunner, arr, index: int):
        name = runner.scene.name
        print("running scene", name)
        success = runner.run()
        # self.results[runner.scene.name] = success
        print("finished running", name, "with outcome", success)

        if not success:
            # write to disk
            runner.flush_to_disk(f"./results/test_suite_{runner.scene.name}.json")

        arr[index] = success

    def run_all_parallel(self):
        self.clear_previous_results()

        results = Array("b", range(len(self.runners)))

        processes = []
        for i, runner in enumerate(self.runners):
            process = Process(
                target=self.run_scene,
                args=(
                    runner,
                    results,
                    i,
                ),
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        print("all tests done")
        results: List[bool] = results[:]  # type: ignore

        for i, result in enumerate(results):
            text = "[success]" if result == 1 else "[fail]   "
            print(f"{text}: {self.runners[i].scene.name}")

    def clear_previous_results(self):
        files = glob.glob("./results/test_suite*.json")
        print("previous runs are", files)

        for file in files:
            os.remove(file)
        # delete all with prefix of ./results/test_suite_***.json
        pass
