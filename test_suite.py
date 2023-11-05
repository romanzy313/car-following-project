import random
from models import ModelV1, RemoteControlled
from src.scene import Scene, make_equadistent_scene
from src.test_runner import TestRunner
from src.vehicle import Vehicle


def main():
    test_runner = TestRunner(False)

    remote_vehicle = Vehicle(
        length=1,
        max_acceleration=10000,
        max_deceleration=10000,
        max_velocity=30000,
        display="blue",
    )
    ai_vehicle = Vehicle(
        length=1, max_acceleration=3, max_deceleration=3, max_velocity=30
    )

    def make_remote_model(initial_position: float, inital_velocity: float, cb):
        remote_model = RemoteControlled.Definition(
            id="Remote",
            vehicle=remote_vehicle,
            initial_position=initial_position,
            inital_velocity=inital_velocity,
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
    slowdown_cb = make_linear_controller(10, 200, 250, 3)
    remote_model = make_remote_model(40, 10, slowdown_cb)

    # this will test the AI model

    ai_model = make_ai_model("1", 0, 10, "./src/model_scaler_cluster_1.pth")
    ai_model2 = make_ai_model("2", 10, 10, "./src/model_scaler_cluster_1.pth")

    test_runner.add_scene(
        Scene(
            name="rapid_breaking",
            models=[ai_model, ai_model2, remote_model],
            road_length=60,
            max_iterations=1000,
        )
    )

    slowdown_cb = make_linear_controller(10, 200, 250, 3)
    remote_model = make_remote_model(40, 10, slowdown_cb)

    # this will test the AI model

    ai_model3 = make_ai_model("1", 10, 10, "./src/model_scaler_cluster_1.pth")
    ai_model4 = make_ai_model("2", 20, 11, "./src/model_scaler_cluster_1.pth")

    test_runner.add_scene(
        Scene(
            name="random_bs",
            models=[ai_model3, ai_model4],
            road_length=40,
            max_iterations=2000,
        )
    )

    test_runner.run_all_parallel()


if __name__ == "__main__":
    random.seed(69)
    main()
