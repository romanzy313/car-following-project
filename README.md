# Difference in car-following interaction between following a human-driven vehicle and an autonomous vehicle project

Short description here

## Gettign started

Install pipenv if not already installed.

```sh
pip install pipenv
```

Install all packages from `Pipfile.lock`, this will take a while.

```sh
pipenv install -d
```

Enter subshell

```sh
pipenv shell
```

Run simple example. It will solve simple simulation and run visualization of the result

```sh
python example_run.py && python visualization.py --file ./results/test_run.json
```

## Rebuilding al training data

This repo contains a trained model of 5 different datasets (2 HA, 2 HH, 1 AH) clusters.
Re-clustering and re-training can be performed by running `sh ./train.sh`

## Run tests

There are a lot of tests to verify that the model works as expected. These tests can be run with the following command:

`pytest -n 2 tests.py`

Running the test will output many simulations in `test_results` folder. They can be visualized inidividually by running

`python visualization.py -f ./test_results/....json`

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

# PYPY SETUP

get BLAS for numpy

`sudo apt-get install libblas-dev liblapack-dev`
`sudo apt-get install gfortran`

`pypy3 -m pip install numpy pandas zarr tqdm`
`pypy3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu117`
