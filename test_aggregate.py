import random
from src.aggregate_runner import AggregateSimulationRunner
from src.vehicle import Vehicle


def main():
    runner = AggregateSimulationRunner(
        dt=0.1,
        model_a_name="RandomAcceleration",
        model_a_args={"spread": 0.3},
        model_b_name="DumbGasser",
        model_b_args={"gas_amount": 0.2},
        initial_velocity=1,
        max_iterations_per_run=5000,
        road_length=40,
        vehicle_count=5,
        scenario_iterations=20,
        sweep_step=0.1,
    )

    runner.run_all()

    runner.flush_to_disk("results/test_aggregate.json")


if __name__ == "__main__":
    main()