import random
from src.aggregate_runner import AggregateSimulationRunner
from src.vehicle import Vehicle


def main():
    runner = AggregateSimulationRunner(
        dt=0.1,
        model_a_name="RandomAcceleration",
        model_a_args={"spread": 0.03},
        # model_b_name="DumbGasser",
        # model_b_args={"gas_amount": 0.01},
        model_b_name="ModelV1",
        model_b_args={
            "model_type": "H",
            "data_file": "./src/model_scaler_cluster_0.pth",
        },
        initial_velocity=5,
        max_iterations_per_run=1000,
        road_length=80,
        vehicle_count=5,
        scenario_iterations=1,
        sweep_step=0.1,
    )

    runner.run_all(pool_size=2)

    runner.flush_to_disk("results/test_aggregate_2.json")


if __name__ == "__main__":
    main()
