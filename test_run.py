from src.model import get_model_from_name
from src.scene import SimulationRunner, make_equadistent_scene
from src.vehicle import Vehicle


def main():
    # get_model_from_name("KeepDistanceModel")
    vehicle = Vehicle(length=1, max_acceleration=1, max_deceleration=-1)

    scene = make_equadistent_scene(
        model_name="KeepDistance",
        model_args={"keep_distance": 5},
        vehicle=vehicle,
        road_length=40,
        vehicle_count=4,
        initial_velocity=1,
        fuzzy_position=1,
    )

    runner = SimulationRunner(scene=scene)

    runner.run(dt=0.1, max_iterations=1000)

    runner.flush_to_disk("results/test_run.json")


if __name__ == "__main__":
    main()
