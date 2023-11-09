#!/bin/bash

pipenv shell
cd ai
python cluster.py
python train.py
