import random
from src.scene import make_equadistent_scene
from src.simulation_runner import SimulationRunner
from src.vehicle import Vehicle


def main():
    dt = 0.1
    max_iterations = 2000
    # get_model_from_name("KeepDistanceModel")
    vehicle = Vehicle(length=1, max_acceleration=1, max_deceleration=1, max_velocity=50)

    scene = make_equadistent_scene(
        dt=dt,
        # model_name="RandomAcceleration",
        # model_args={"spread": 0.1},
        model_name="ModelV1",
        model_args={"model_type": "H", "data_file": "./src/model_scaler_cluster_0.pth"},
        vehicle=vehicle,
        road_length=200,
        vehicle_count=10,
        initial_velocity=15,
        fuzzy_position=0.5,
        max_iterations=max_iterations,
    )

    runner = SimulationRunner(scene, True, True, True)

    print("running...")

    runner.run()

    runner.flush_to_disk("./results/test_run.json")


if __name__ == "__main__":
    random.seed(69)
    main()
