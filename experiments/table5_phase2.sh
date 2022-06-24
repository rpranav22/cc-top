#!/bin/bash
# bash script to repeat best model over folds

CONFIG="dbpedia_td_p2.yaml"

# give the child a name
NAME='DBPtd_01_p2_3v1'
RUN="kcl_5k_01_3v1"

sbatch --partition=mcml-dgx-a100-40x8 --time=01-00:00:00 --job-name=$NAME --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME --error=/home/di39tih2/cc-top/error/$NAME -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name $RUN