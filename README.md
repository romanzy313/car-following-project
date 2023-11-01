# Difference in car-following interaction between following a human-driven vehicle and an autonomous vehicle project

Project description

# Run example simulation

`python test_run.py && python visualization.py`

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
