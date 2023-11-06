from functools import partial
import json
from typing import List


from src.scene import Scene, make_a_b_scene
from src.vehicle import Vehicle
from multiprocessing import Pool, cpu_count


def xfrange(start, stop, step):
    i = 0
    while start + i * step <= stop:
        yield start + i * step
        i += 1


class AggregateSimulationRunner:
    """
    This gives the ability to run simulation with 2 models, slowly increasing them
    """

    # sample means how many times thi
    def __init__(
        self,
        dt: float,
        model_a_name: str,
        model_b_name: str,
        initial_velocity: float,
        road_length: float,
        vehicle_count: int,
        sweep_step: float,  # how much should the percentage of a be increased by, upto 1
        max_iterations_per_run: int,  # how many iterations in each run
        scenario_iterations: int,  # how many runs are performed for statistical certanty
        model_a_args={},
        model_b_args={},
    ) -> None:
        self.dt = dt
        self.max_iterations_per_run = max_iterations_per_run
        self.scenario_iterations = scenario_iterations
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.initial_velocity = initial_velocity
        self.road_length = road_length
        self.vehicle_count = vehicle_count
        self.sweep_step = sweep_step
        self.model_a_args = model_a_args
        self.model_b_args = model_b_args

        self.vehicle = Vehicle(
            length=1,
            max_acceleration=1,
            max_deceleration=1,
            max_velocity=20,
        )

        self.results = []

    def single_run(
        self, sweep_amount: float, scenario_id: int, run_id: int, total_count: int
    ):
        print(
            f"[{run_id}/{total_count}] scenario {scenario_id} with sweep {sweep_amount}"
        )
        # Make a scene
        scene = make_a_b_scene(
            model_a_name=self.model_a_name,
            model_b_name=self.model_b_name,
            model_a_args=self.model_a_args,
            model_b_args=self.model_b_args,
            max_iterations=self.max_iterations_per_run,
            dt=self.dt,
            a_percentage=sweep_amount,
            initial_velocity=self.initial_velocity,
            road_length=self.road_length,
            vehicle_count=self.vehicle_count,
            vehicle=self.vehicle,
            random_seed=run_id + 420,
        )

        results = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "model_a": self.model_a_name,
            "model_b": self.model_b_name,
            "sweep_amount": round(sweep_amount, 2),
        }

        # run-it
        run_res = scene.run(
            with_steps=False, with_statistics=True, display_progress=False
        )

        # print(f"[{run_id}] COMPLETED scenario {scenario_id} with sweep {sweep_amount}")

        return {**results, **run_res}

    def run_all(self, pool_size: int = cpu_count()):
        run_id = 0
        scenario_id = 0
        # give them total count
        desired_runs = []
        total_count = (round(1 / self.sweep_step) + 1) * self.scenario_iterations
        # print("total count is", total_count)

        for sweep_amount in xfrange(0, 1, self.sweep_step):
            scenario_id += 1
            for _ in range(0, self.scenario_iterations):
                run_id += 1
                desired_runs.append(
                    (round(sweep_amount, 2), scenario_id, run_id, total_count)
                )

        # first build up an array of values that must be run
        # then somehow
        # now make it use the pool

        proc_pool = Pool(pool_size)

        self.results = proc_pool.starmap(self.single_run, desired_runs)

        proc_pool.close()
        proc_pool.join()

        print("aggregate runner finished")

        pass

    def flush_to_disk(self, file: str):
        print(f"writing aggregate simulation output to {file}")

        with open(f"{file}", "w") as fp:
            json.dump(
                self.results,
                fp,
                indent=2,
            )
