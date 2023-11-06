# Difference in car-following interaction between following a human-driven vehicle and an autonomous vehicle project

## Run tests

`pytest -n 2 tests.py`

### only a subset of tests

`pytest -n 2 -k manual ./tests.py`

## Visualize output

`python visualization.py -f ./test_results/deceleration_10_4_in_3.json`

Project description

# Run example simulation

`python test_run.py && python visualization.py`

# More Notes

- Test proportion of different models (create 9)
- 2 clusters for HA. Human vs Human.
- Humans vs Autonomous
- H-A-H-A-H-A (50% H, 50% A)
- H-A-A-H-A-A (33% H, 60% A)
- Can do 4 models (Ha, Hb, Hc, A)

# Notes

- Perform inital dimension reduction on 3 second intervals of the training set.
- Run simulation multiple times with different starting configurations
- Use different reaction models (based on classification), using case_id, and resulting models are denoted as f1, f2, ... ,fn
- Each vehicle can have its own
- For clustering: Use k-means or unsupervised clustering, aggregate this data by aggressiveness (minimum Time To Collision): minTTC = (delta position)/(delta time)
- Other useful features maxA, maxV, std of anything. etc...
- Then do simulation by selecting populations of data
- Set acceleration as the output variable

# Tutorials

- Enter virtual environment with `pipenv shell`
- Install packages via `pipenv install ...`

# How to use

- run with `python cli.py -s ./scenes/test.json`

# Errors and fixes

- Linux amd driver fix `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`

Test_change

## meeting notes

- Learning needs to take place over all the data, shifted by a single timestamp
- Aggregated data only used for clustering
- 9 Models for car following. HA, AA, AH, each has 3 clusters.
- input: velocity, delta_velocity, delta_position... Give 30 steps. Make sure its output is correct for the next 10 steps (which is also v, delta_v, delta_p)
- output: velocity (non-recommended) or acceleration
- use 10 steps during runtime, learn with 10 steps aswell

- we give AI delta_velocity, delta_position, initial_velocity. The output is (delta_velocity, delta_position, initial_velocity)
