#!/bin/bash

pipenv shell
cd ai
python cluster.py
python train_v0.py