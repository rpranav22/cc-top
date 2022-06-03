#!/bin/bash
# bash script to repeat best model over folds

# Specify the config file you want to tune
CONFIG="table1_dbpedia_constraint.yaml"

# give the child a name
NAME='DPBt'

IT=0

for FOLD in 0 1 2 3 4; do
    sbatch --partition=mcml-dgx-a100-40x8 --time=01-00:00:00 --job-name=$NAME-$FOLD --qos=mcml --output=/home/di39tih2/out/$NAME-$FOLD --error=/home/di39tih2/cc-top/error/$NAME-$FOLD -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG
    if (( $IT == 0)); then
        sleep 60s
    fi
    ((IT+=1))
    sleep 10s
done