from src.model import KeepDistanceModel
from src.scene import SimulationRunner, make_equadistent_scene
from src.vehicle import Vehicle


def main():
    vehicle = Vehicle(length=1, max_acceleration=1, max_deceleration=-1)

    scene = make_equadistent_scene(
        vehicle=vehicle,
        road_length=40,
        vehicle_count=4,
        initial_velocity=1,
        fuzzy_position=1,
        model=KeepDistanceModel,
        model_args={"keep_distance": 5},
    )

    runner = SimulationRunner(scene=scene)

    runner.run(dt=0.1, max_iterations=1000)

    runner.flush_to_disk("results/test_run.json")


if __name__ == "__main__":
    main()
