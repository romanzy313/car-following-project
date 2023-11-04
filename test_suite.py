import random
from models import ModelV1, RemoteControlled
from src.scene import Scene, make_equadistent_scene
from src.test_runner import TestRunner
from src.vehicle import Vehicle


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


# # callbacks, always give it 200 steps
# def gradual_raise(iteration):
#     if iteration <= 200:
#         return 2  # Output is 2 for the first 100 iterations
#     elif iteration <= 300:
#         # Gradually increase output from 2 to 10 between iterations 100 and 200
#         progress = (iteration - 100) / 100.0  # Progress from 0 to 1
#         return 2 + progress * 8  # Linear interpolation from 2 to 10
#     else:
#         return 10  # Output is 10 after the 200th iteration


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
    ai_model2 = make_ai_model("1", 10, 10, "./src/model_scaler_cluster_1.pth")

    test_runner.add_scene(
        Scene(
            name="first one",
            models=[ai_model, ai_model2, remote_model],
            road_length=60,
            max_iterations=1000,
        )
    )

    slowdown_cb = make_linear_controller(10, 200, 250, 3)
    remote_model = make_remote_model(40, 10, slowdown_cb)

    # this will test the AI model

    ai_model3 = make_ai_model("1", 10, 10, "./src/model_scaler_cluster_1.pth")
    ai_model4 = make_ai_model("1", 20, 11, "./src/model_scaler_cluster_1.pth")

    test_runner.add_scene(
        Scene(
            name="second one",
            models=[ai_model3, ai_model4],
            road_length=40,
            max_iterations=2000,
        )
    )

    # scene = make_equadistent_scene(
    #     dt=dt,
    #     # model_name="RandomAcceleration",
    #     # model_args={"spread": 0.1},
    #     model_name="ModelV1",
    #     model_args={"model_type": "H", "data_file": "./src/model_scaler_cluster_0.pth"},
    #     vehicle=vehicle,
    #     road_length=200,
    #     vehicle_count=10,
    #     initial_velocity=3,
    #     fuzzy_position=1,
    #     max_iterations=max_iterations,
    # )

    print("running experiment speed up")

    test_runner.run_all()

    # runner = SimulationRunner(scene=scene, dt=dt, max_iterations=max_iterations)

    # runner.run()

    # print("starting to flush to disk...")

    # runner.flush_to_disk("results/test_suite.json")


if __name__ == "__main__":
    random.seed(69)
    main()
