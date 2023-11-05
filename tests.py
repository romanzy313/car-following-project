from typing import List
import unittest

import pytest
from models import ModelV1, RemoteControlled
from src.model import Model
from src.scene import Scene, make_equadistent_scene
from src.simulation_runner import SimulationRunner
from src.vehicle import Vehicle

remote_vehicle = Vehicle(
    length=1,
    max_acceleration=10000,
    max_deceleration=10000,
    max_velocity=30000,
    display="blue",
)
ai_vehicle = Vehicle(length=1, max_acceleration=1, max_deceleration=1, max_velocity=20)


def combine_controllers(otherwise: float, controllers_def):
    pass


# always start after
def make_linear_controller(
    start: float, start_iteration: int, end_iteration: int, final: float
):
    def run(iteration: int):
        if iteration <= start_iteration:
            return start
        elif iteration <= end_iteration:
            progress = (iteration - start_iteration) / (
                end_iteration - start_iteration
            )  # Progress from 0 to 1
            # print("progress", progress)
            return start + progress * (final - start)
        else:
            return final

    return run


def make_constant_controller(value: float):
    def run(iteration: int):
        return value

    return run


def make_remote_model(position: float, cb):
    remote_model = RemoteControlled.Definition(
        id="Remote",
        vehicle=remote_vehicle,
        initial_position=position,
        inital_velocity=cb(0),
    )
    remote_model.set_callback(cb)
    return remote_model


def make_ai_model(
    index: str, initial_position: float, inital_velocity: float, data_file: str
):
    ai_model = ModelV1.Definition(
        id=f"AI{index}",
        vehicle=ai_vehicle,
        initial_position=initial_position,
        inital_velocity=inital_velocity,
    )
    ai_model.inject_args({"model_type": "H", "data_file": data_file})
    return ai_model


def make_many_ai_models(
    count: int,
    start_position: float,
    end_position: float,
    inital_velocity: float,
    data_file: str,
):
    results: List[Model] = []
    spacing = (end_position - start_position) / count
    for i in range(count):
        pos = start_position + i * spacing
        results.append(make_ai_model(str(i), pos, inital_velocity, data_file))

    return results


class TestDrive:
    # @pytest.mark.skip()
    @pytest.mark.parametrize("speed", [1, 5, 10, 15])
    def test_constant_speed(self, speed):
        # check that car can drive straight without colliding

        remote_model = make_remote_model(5, make_constant_controller(speed))
        ai_model = make_ai_model("1", 0, speed, "./src/model_scaler_cluster_1.pth")

        runner = SimulationRunner(
            Scene(
                name=f"constant_speed_{speed}",
                models=[ai_model, remote_model],
                road_length=100,
                max_iterations=500,
            )
        )

        runner.run()
        runner.flush_to_disk()

        assert runner.did_collide() == False, "collision"

    @pytest.mark.parametrize("speed,min_speed,max_speed", [(1, 0.5, 1.5), (10, 9, 11)])
    def test_single_keep_speed(self, speed, min_speed, max_speed):
        # check that car can drive straight without colliding

        ai_model = make_ai_model("1", 0, speed, "./src/model_scaler_cluster_1.pth")

        runner = SimulationRunner(
            Scene(
                name=f"single_vehicle_{speed}",
                models=[ai_model],
                road_length=100,
                max_iterations=300,
            ),
        )

        runner.run()
        runner.flush_to_disk()
        final_vel = runner.scene.models[0].velocities[-1]
        assert final_vel > min_speed, f"too slow {final_vel}"
        assert final_vel < max_speed, f"too fast {final_vel}"

    # @pytest.mark.skip()
    @pytest.mark.parametrize(
        "start_speed,end_speed,time",
        [(10, 8, 4.0), (10, 6, 3.0), (10, 4, 3), (10, 0, 10)],
    )
    def test_rapid_break(self, start_speed: float, end_speed: float, time: float):
        timesteps = round(time / 0.1)
        remote_model = make_remote_model(
            60, make_linear_controller(start_speed, 200, 200 + timesteps, end_speed)
        )
        ai_model1 = make_ai_model(
            "1", 0, start_speed, "./src/model_scaler_cluster_1.pth"
        )
        ai_model2 = make_ai_model(
            "2", 20, start_speed, "./src/model_scaler_cluster_1.pth"
        )
        ai_model3 = make_ai_model(
            "3", 40, start_speed, "./src/model_scaler_cluster_1.pth"
        )

        runner = SimulationRunner(
            Scene(
                name=f"deceleration_{start_speed}_{end_speed}_in_{time}",
                models=[ai_model1, ai_model2, ai_model3, remote_model],
                road_length=100,
                max_iterations=800,
            )
        )

        runner.run()
        runner.flush_to_disk()

        assert runner.did_collide() == False, "collision"

    # @pytest.mark.skip()
    @pytest.mark.parametrize(
        "vehicle_count,initial_velocity,low_limit,high_limit",
        [(10, 2, 1, 3), (10, 5, 4, 6), (10, 10, 9, 11)],
    )
    def test_many_equadistent(
        self,
        vehicle_count: int,
        initial_velocity: float,
        low_limit: float,
        high_limit: float,
    ):
        scene = make_equadistent_scene(
            scene_name=f"many_{vehicle_count}_at_speed_{initial_velocity}",
            model_name="ModelV1",
            model_args={
                "model_type": "H",
                "data_file": "./src/model_scaler_cluster_0.pth",
            },
            vehicle=ai_vehicle,
            road_length=100,
            vehicle_count=vehicle_count,
            initial_velocity=initial_velocity,
            fuzzy_position=0.5,
            max_iterations=1000,
        )
        runner = SimulationRunner(scene, with_statistics=True)
        runner.run()
        runner.flush_to_disk()
        assert runner.did_collide() == False, "collision"
        avg_vel = runner.get_result_value("velocity_mean")

        assert avg_vel > low_limit, f"too slow {avg_vel}"
        assert avg_vel < high_limit, f"too fast {avg_vel}"

    # this one keeps on failing
    @pytest.mark.parametrize(
        "speed,count,min_speed,max_speed",
        [(10, 1, 1, 3), (10, 5, 4, 6), (10, 10, 9, 11)],
    )
    def test_many_manual(self, speed, count, min_speed, max_speed):
        # remote_model = make_remote_model(80, make_constant_controller(speed))
        ai_models = make_many_ai_models(
            count, 0, 60, speed, "./src/model_scaler_cluster_1.pth"
        )
        runner = SimulationRunner(
            Scene(
                name=f"many_count_{count}_at_speed_{speed}",
                # models=[*ai_models, remote_model],
                models=[*ai_models],
                road_length=100,
                max_iterations=1000,
            ),
            with_statistics=True,
        )

        runner.run()
        runner.flush_to_disk()
        assert runner.did_collide() == False, "collision"

        avg_vel = runner.get_result_value("velocity_mean")

        assert avg_vel > min_speed, f"too slow {avg_vel}"
        assert avg_vel < max_speed, f"too fast {avg_vel}"
