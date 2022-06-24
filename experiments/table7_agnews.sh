#!/bin/bash
# bash script to repeat best model over folds

# Specify the config file you want to tune
CONFIG="agnews_constrained.yaml"

# give the child a name
NAME1='AGNcc1_MCL'

RUN1="mcl_1k"

IT=0

for FOLD in 0 1 2 3 4; do
    sbatch --partition=mcml-dgx-a100-40x8 --time=01-00:00:00 --job-name=$NAME1-$FOLD --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME1-$FOLD --error=/home/di39tih2/cc-top/error/$NAME1-$FOLD -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name $RUN1-$FOLD --num_constraints 1000 --max_epochs 200
    if (( $IT == 0)); then
        sleep 60s
    fi
    ((IT+=1))
    sleep 10s
done

# give the child a name
NAME1='AGNcc5_MCL'

RUN1="mcl_5k"

IT=0

for FOLD in 0 1 2 3 4; do
    sbatch --partition=mcml-dgx-a100-40x8 --time=01-00:00:00 --job-name=$NAME1-$FOLD --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME1-$FOLD --error=/home/di39tih2/cc-top/error/$NAME1-$FOLD -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name $RUN1-$FOLD --num_constraints 5000 --max_epochs 200
    if (( $IT == 0)); then
        sleep 60s
    fi
    ((IT+=1))
    sleep 10s
done


# # give the child a name
# NAME1='AGNcc10_MCL'

# RUN1="mcl_10k"

# IT=0

# for FOLD in 0 1 2 3 4; do
#     sbatch --partition=mcml-dgx-a100-40x8 --time=01-00:00:00 --job-name=$NAME1-$FOLD --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME1-$FOLD --error=/home/di39tih2/cc-top/error/$NAME1-$FOLD -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name $RUN1-$FOLD --num_constraints 10000 --max_epochs 200
#     if (( $IT == 0)); then
#         sleep 60s
#     fi
#     ((IT+=1))
#     sleep 10s
# done

# give the child a name
NAME1='AGNcc20_MCL'

RUN1="mcl_20k"

IT=0

for FOLD in 0 1 2 3 4; do
    sbatch --partition=mcml-dgx-a100-40x8 --time=01-00:00:00 --job-name=$NAME1-$FOLD --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME1-$FOLD --error=/home/di39tih2/cc-top/error/$NAME1-$FOLD -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name $RUN1-$FOLD --num_constraints 20000 --max_epochs 200
    if (( $IT == 0)); then
        sleep 60s
    fi
    ((IT+=1))
    sleep 10s
done

# give the child a name
NAME1='AGNcc30_MCL'

RUN1="mcl_30k"

IT=0

for FOLD in 0 1 2 3 4; do
    sbatch --partition=mcml-dgx-a100-40x8 --time=01-00:00:00 --job-name=$NAME1-$FOLD --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME1-$FOLD --error=/home/di39tih2/cc-top/error/$NAME1-$FOLD -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name $RUN1-$FOLD --num_constraints 30000 --max_epochs 200
    if (( $IT == 0)); then
        sleep 60s
    fi
    ((IT+=1))
    sleep 10s
done