#!/bin/bash
# bash script to repeat best model over folds

# Specify the config file you want to tune
CONFIG="trec_constrained.yaml"

# give the child a name
NAME1='TRCoc1'

RUN1="kcl_1k_100c"

IT=0

for FOLD in 0 1 2 3 4; do
    sbatch --partition=mcml-dgx-a100-40x8 --time=01-00:00:00 --job-name=$NAME1-$FOLD --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME1-$FOLD --error=/home/di39tih2/cc-top/error/$NAME1-$FOLD -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name $RUN1-$FOLD --num_constraints 1000 --max_epochs 300
    if (( $IT == 0)); then
        sleep 60s
    fi
    ((IT+=1))
    sleep 10s
done

# # give the child a name
# NAME2='TRCoc2'

# RUN2="kcl_5k_100c"

# IT=0

# for FOLD in 0 1 2 3 4; do
#     sbatch --partition=mcml-dgx-a100-40x8 --time=01-00:00:00 --job-name=$NAME2-$FOLD --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME2-$FOLD --error=/home/di39tih2/cc-top/error/$NAME2-$FOLD -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name $RUN2-$FOLD --num_constraints 5000 --max_epochs 200
#     if (( $IT == 0)); then
#         sleep 60s
#     fi
#     ((IT+=1))
#     sleep 10s
# done

# # give the child a name
# NAME3='TRCoc3'

# RUN3="kcl_10k_100c"

# IT=0

# for FOLD in 0 1 2 3 4; do
#     sbatch --partition=mcml-dgx-a100-40x8 --time=01-00:00:00 --job-name=$NAME3-$FOLD --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME3-$FOLD --error=/home/di39tih2/cc-top/error/$NAME3-$FOLD -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name $RUN3-$FOLD --num_constraints 10000 --max_epochs 200
#     if (( $IT == 0)); then
#         sleep 60s
#     fi
#     ((IT+=1))
#     sleep 10s
# done

# # give the child a name
# NAME3='TRCoc4'

# RUN3="kcl_20k_100c"

# IT=0

# for FOLD in 0 1 2 3 4; do
#     sbatch --partition=mcml-dgx-a100-40x8 --time=01-00:00:00 --job-name=$NAME3-$FOLD --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME3-$FOLD --error=/home/di39tih2/cc-top/error/$NAME3-$FOLD -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name $RUN3-$FOLD --num_constraints 20000 --max_epochs 200
#     if (( $IT == 0)); then
#         sleep 60s
#     fi
#     ((IT+=1))
#     sleep 10s
# done


# # give the child a name
# NAME4='TRCoc5'

# RUN4="kcl_30k_100c"

# IT=0

# for FOLD in 0 1 2 3 4; do
#     sbatch --partition=mcml-dgx-a100-40x8 --time=01-00:00:00 --job-name=$NAME4-$FOLD --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME4-$FOLD --error=/home/di39tih2/cc-top/error/$NAME4-$FOLD -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name $RUN4-$FOLD --num_constraints 30000 --max_epochs 200
#     if (( $IT == 0)); then
#         sleep 60s
#     fi
#     ((IT+=1))
#     sleep 10s
# done