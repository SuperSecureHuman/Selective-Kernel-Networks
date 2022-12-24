#!/bin/bash

NUM=$1

python train.py --learning_rate=0.0005 --batch_size=16 --epochs=20 --model=SKNet$1 --wandb=True
python train.py --learning_rate=0.0001 --batch_size=16 --epochs=20 --model=SKNet$1 --wandb=True
python train.py --learning_rate=0.001 --batch_size=16 --epochs=20 --model=SKNet$1 --wandb=True
