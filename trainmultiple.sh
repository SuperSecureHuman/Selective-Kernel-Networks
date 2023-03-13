#!/bin/bash

python train.py --learning_rate=0.0005 --batch_size=16 --epochs=30 --model=SKNet101 --wandb=True --data=$1 --seed=10 --image_size=256
sleep 20
python train.py --learning_rate=0.0001 --batch_size=16 --epochs=30 --model=SKNet101 --wandb=True --data=$1 --seed=10 --image_size=256
sleep 20
python train.py --learning_rate=0.001 --batch_size=16 --epochs=30  --model=SKNet101 --wandb=True --data=$1 --seed=10 --image_size=256
