from src.scene import SimulationRunner, make_equadistent_scene
from src.vehicle import Vehicle


def main():
    vehicle = Vehicle(id="1", length=1, max_acceleration=1, max_deceleration=1)

    scene = make_equadistent_scene(
        vehicle=vehicle, road_length=40, vehicle_count=4, initial_velocity=1
    )

    runner = SimulationRunner(scene=scene, dt=0.1, max_iterations=1000)

    runner.run()

    runner.flush_to_disk("results/test.json")


if __name__ == "__main__":
    main()
