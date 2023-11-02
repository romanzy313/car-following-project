import random
from src.scene import make_equadistent_scene
from src.simulation_runner import SimulationRunner
from src.vehicle import Vehicle


def main():
    dt = 0.1
    max_iterations = 1000
    # get_model_from_name("KeepDistanceModel")
    vehicle = Vehicle(length=1, max_acceleration=1, max_deceleration=1, max_velocity=30)

    scene = make_equadistent_scene(
        dt=dt,
        model_name="RandomAcceleration",
        model_args={"spread": 0.3},
        vehicle=vehicle,
        road_length=40,
        vehicle_count=5,
        initial_velocity=1,
        fuzzy_position=1,
    )

    runner = SimulationRunner(scene=scene, dt=dt, max_iterations=max_iterations)

    runner.run()

    runner.flush_to_disk("results/test_run.json")


if __name__ == "__main__":
    random.seed(69)
    main()
