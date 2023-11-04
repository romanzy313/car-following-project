from typing import Any, List
import multiprocessing
from src.scene import Scene

from src.simulation_runner import SimulationRunner


class TestRunner:
    runners: List[SimulationRunner]
    results: dict

    def __init__(self, quick_fail: bool) -> None:
        self.quick_fail = quick_fail
        self.runners = []
        pass

    def add_scene(
        self,
        scene: Scene,
    ):
        self.runners.append(SimulationRunner(scene))

    def run_scene(self, runner: SimulationRunner, result_queue):
        name = runner.scene.name
        print("running scene", name)
        success = runner.run(True, False)
        # self.results[runner.scene.name] = success
        print("run result", runner.scene.name, success)

        if not success:
            # write to disk
            runner.flush_to_disk(f"./results/test_{runner.scene.name}.json")

        result_queue.put((runner, success))

        return 1

    def run_all(self):
        result_queue = multiprocessing.Queue()
        processes = []
        # use parallel

        for runner in self.runners:
            process = multiprocessing.Process(
                target=self.run_scene,
                args=(runner, result_queue),
            )
            processes.append(process)
            process.daemon = False
            # process.run()
            process.start()

        for process in processes:
            process.join()

        print("HERE")

        # Collect results in the original order
        results_dict = {}
        while not result_queue.empty():
            print("not empty")
            runner, success = result_queue.get()
            name = runner.scene.name
            print("got scene", name, "outcome", success)

            # if not success:
            #     # write to disk
            #     runner.flush_to_disk(f"test_{name}")

            results_dict[runner] = success

        print("EMPTY")

        # self.results = [results_dict[scene] for scene in self.runners]

        # print("results are", self.results)
