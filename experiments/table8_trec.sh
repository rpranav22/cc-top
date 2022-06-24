#!/bin/bash
# bash script to repeat best model over folds

# Specify the config file you want to tune
CONFIG="trec_constrained.yaml"

# give the child a name
NAME3='TRCoc3-100'

RUN3="mcl_10k_100c"

IT=0

for FOLD in 0 1 2 3 4; do
    sbatch --partition=mcml-dgx-a100-40x8 --time=01-00:00:00 --job-name=$NAME3-$FOLD --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME3-$FOLD --error=/home/di39tih2/cc-top/error/$NAME3-$FOLD -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name $RUN3-$FOLD --num_constraints 10000 --max_epochs 200
    if (( $IT == 0)); then
        sleep 60s
    fi
    ((IT+=1))
    sleep 10s
done