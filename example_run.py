import random
from src.scene import make_equadistent_scene
from src.simulation_runner import SimulationRunner
from src.vehicle import Vehicle


def main():
    dt = 0.1
    max_iterations = 800
    vehicle = Vehicle(length=1, max_acceleration=1, max_deceleration=1, max_velocity=30)

    scene = make_equadistent_scene(
        scene_name="example",
        model_name="ModelV1",
        model_args={
            "model_type": "H",
            "data_file": "./out_brain_64/AH_0.pth",
        },
        # model_args={"model_type": "H", "data_file": "./src/model_scaler_cluster_1.pth"},
        vehicle=vehicle,
        road_length=200,
        vehicle_count=10,
        initial_velocity=15,
        fuzzy_position=3,
        max_iterations=max_iterations,
    )

    runner = SimulationRunner(scene, True, True, True)

    runner.run()

    runner.flush_to_disk_to_file("./results/test_run.json")


if __name__ == "__main__":
    random.seed(69)
    main()
