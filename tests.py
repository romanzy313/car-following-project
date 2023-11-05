import unittest
from models import ModelV1, RemoteControlled
from src.scene import Scene, make_equadistent_scene
from src.simulation_runner import SimulationRunner
from src.test_runner import TestRunner
from src.vehicle import Vehicle

remote_vehicle = Vehicle(
    length=1,
    max_acceleration=10000,
    max_deceleration=10000,
    max_velocity=30000,
    display="blue",
)
ai_vehicle = Vehicle(length=1, max_acceleration=3, max_deceleration=3, max_velocity=30)


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


# need callbacks for this
# slowdown_cb = make_linear_controller(10, 200, 250, 3)


class TestDrive(unittest.TestCase):
    def test_slow_speed(self):
        # check that car can drive straight without colliding

        remote_model = make_remote_model(10, make_constant_controller(10))
        ai_model = make_ai_model("1", 0, 10, "./src/model_scaler_cluster_1.pth")

        runner = SimulationRunner(
            Scene(
                name="contant_follow",
                models=[ai_model, remote_model],
                road_length=20,
                max_iterations=500,
            )
        )

        runner.run()
        runner.save_test("slow_speed")

        self.assertEqual(runner.did_collide(), False, "collision detected")
        self.assertLessEqual(runner.get_position_with_first(), 10, "strayed too far")

        pass
