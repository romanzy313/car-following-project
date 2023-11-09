import src.patch_istarmap
from functools import partial
import json
import random
from typing import List

from tqdm import tqdm
from models import ModelV1
from src.model import Model


from src.scene import Scene
from src.vehicle import Vehicle
from multiprocessing import Pool
import re


def extract_unique_dateset_cluster(brain_file: str) -> str:
    match = re.match(r".*?/([AH|HA|HH]+)_([0-9]+)\.pth", brain_file)
    if match:
        dataset_name, cluster = match.groups()
        cluster = int(cluster)
        return f"{dataset_name}_{cluster}"

    raise Exception(f"failed to parse brain file {brain_file}")
    # return None


def xfrange(start, stop, step):
    i = 0
    while start + i * step <= stop:
        yield start + i * step
        i += 1


class AggregateRunner:
    def get_randomized_brains(self, ai_percent: float) -> List[str]:
        ai_drivers = round(ai_percent * self.vehicle_count)
        human_drivers = round((1 - ai_percent) * self.vehicle_count)
        ai_brains = random.choices(self.ai_brains, k=ai_drivers)
        human_brains = random.choices(self.human_brains, k=human_drivers)
        result = [*ai_brains, *human_brains]
        random.shuffle(result)
        return result

    def create_model(self, id: str, brain: str, position: float, velocity: float):
        model = ModelV1.Definition(
            id=id,
            vehicle=self.vehicle,
            initial_position=position,
            inital_velocity=velocity,
        )
        extracted = extract_unique_dateset_cluster(brain)
        model.inject_args({"model_type": extracted, "data_file": brain})
        return model

    def create_scene(
        self, run_id: int, ai_percent: float, fuzzy_pos=0.2, fuzzy_vel=0.05
    ):
        random.seed(1337 + run_id)
        brains = self.get_randomized_brains(ai_percent)
        spacing = self.road_length / self.vehicle_count

        models: List[Model] = []

        # equadistant put them
        for i in range(self.vehicle_count):
            position = i * spacing

            # a bit fuzzy to avoid infinite ttc and other perfect wierdness
            position_offset = random.uniform(-fuzzy_pos, fuzzy_pos)
            velocity_offset = random.uniform(-fuzzy_vel, fuzzy_vel)

            model = self.create_model(
                str(i),
                brains[i],
                position + position_offset,
                self.initial_velocity + velocity_offset,
            )

            models.append(model)

        return Scene(
            models=models,
            road_length=self.road_length,
            max_iterations=self.max_iterations_per_run,
            name=f"{run_id}",
        )

    # TODO add fixed speed car

    # sample means how many times thi
    def __init__(
        self,
        vehicle: Vehicle,
        ai_brains: List[str],
        human_brains: List[str],
        run_velocity: float,
        road_length: float,
        vehicle_count: int,
        sweep_step: float,  # how much should the percentage of a be increased by, upto 1
        max_iterations_per_run: int,  # how many iterations in each run
        scenario_iterations: int,  # how many runs are performed for statistical certanty
    ) -> None:
        self.max_iterations_per_run = max_iterations_per_run
        self.scenario_iterations = scenario_iterations
        self.ai_brains = ai_brains
        self.human_brains = human_brains
        self.initial_velocity = run_velocity
        self.road_length = road_length
        self.vehicle_count = vehicle_count
        self.sweep_step = sweep_step

        self.vehicle = vehicle

        self.results = []

    def single_run(self, ai_percent: float, scenario_id: int, run_id: int):
        scene = self.create_scene(run_id, ai_percent)

        run_result = scene.run(
            with_steps=False, with_statistics=True, display_progress=False
        )

        # print(
        #     f"[{run_id}] COMPLETED scenario {scenario_id} with ai_population {ai_percent}"
        # )

        return {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "ai_percent": ai_percent,
            **run_result,
        }

    def run_all_parallel(self, worker_size: int = 2):
        run_id = 0
        scenario_id = 0
        # give them total count
        desired_runs = []
        # total_count = (round(1 / self.sweep_step) + 1) * self.scenario_iterations
        # print("total count is", total_count)

        for ai_population in xfrange(0, 1, self.sweep_step):
            scenario_id += 1
            for _ in range(0, self.scenario_iterations):
                run_id += 1
                desired_runs.append((round(ai_population, 2), scenario_id, run_id))

        # first build up an array of values that must be run
        # then somehow
        # now make it use the pool
        proc_pool = Pool(worker_size)

        # inputs = zip(param1, param2, param3)

        # results = pool.starmap(self.sin, tqdm(inputs, total=len(param1)))
        for data in tqdm(
            proc_pool.istarmap(self.single_run, desired_runs), total=len(desired_runs)  # type: ignore
        ):
            # print("data received", data)
            self.results.append(data)
        # self.results = proc_pool.starmap(self.single_run, desired_runs)

        proc_pool.close()
        proc_pool.join()

        print("aggregate runner finished")

        pass

    def flush_json_to_disk(self, file: str):
        print(f"writing aggregate simulation output to {file}")

        with open(f"{file}", "w") as fp:
            json.dump(
                self.results,
                fp,
                indent=2,
            )
