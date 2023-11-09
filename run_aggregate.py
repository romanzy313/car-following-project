from pathlib import Path
from src.aggregate_runner2 import AggregateSimulationRunner2
from src.vehicle import Vehicle


run_velocity = 12
max_iterations_per_run = 1000
road_length = 200
vehicle_count = 10
scenario_iterations = 50
sweep_step = 0.2

ai_brains = ["./out_brain_64/AH_0.pth"]
human_brains = [
    "./out_brain_64/HA_0.pth",
    "./out_brain_64/HA_1.pth",
    "./out_brain_64/HA_2.pth",
    "./out_brain_64/HH_0.pth",
    "./out_brain_64/HH_1.pth",
    "./out_brain_64/HH_2.pth",
    # "./out_brain/HH_1.pth",
]


def main():
    Path("./results").mkdir(parents=False, exist_ok=True)

    ai_vehicle = Vehicle(
        length=1, max_acceleration=1, max_deceleration=1, max_velocity=20
    )
    runner = AggregateSimulationRunner2(
        vehicle=ai_vehicle,
        ai_brains=ai_brains,
        human_brains=human_brains,
        max_iterations_per_run=max_iterations_per_run,
        road_length=road_length,
        run_velocity=run_velocity,
        scenario_iterations=scenario_iterations,
        sweep_step=sweep_step,
        vehicle_count=vehicle_count,
    )

    runner.run_all_parallel(worker_size=2)

    runner.flush_json_to_disk("./results/test_aggregate.json")


if __name__ == "__main__":
    main()
