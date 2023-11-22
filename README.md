# Human Drivers' Reactions to Automated Vehicles: An Analysis on Traffic System Impact

Source code for final project of IFEEMCS520100 Fundamentals of Artificial Intelligence Programme course at TU Delft.

## Authors

- Roman Volovoy (R.Volovoy@student.tudelft.nl)
- Xianzhi Zhang (X.Zhang-85@student.tudelft.nl)
- Yuncong Liu (Y.Liu-138@student.tudelft.nl)
- Zhengchu Wu (Z.Wu-27@student.tudelft.nl)

## Getting started

Pipenv is used for dependency management, install it if needed.

```sh
pip install pipenv
```

Install all packages with dev dependencies from `Pipfile.lock`; this will take a while.

```sh
pipenv install -d
```

Enter python environment

```sh
pipenv shell
```

Run a simple example. It will run an example simulation and then launch a visualization of the result.

```sh
python example_run.py && python visualization.py --file ./results/test_run.json
```

## Rebuilding AI training data

This repo contains trained models of all different datasets (2 HA, 2 HH, 1 AH) clusters.
Re-clustering and re-training can be performed by running `sh ./train.sh`. This process can take a very long time (approx. 2 hours on 12 core machine)

## Run tests

There are many small unit tests to verify that the models work by themselves. These tests can be run with the following command:

`pytest -n 2 tests.py`

Running the tests will output all simulation results in the `test_results` folder. The output can be viewed by running

`python visualization.py -f test_results/<test-name>.json`
