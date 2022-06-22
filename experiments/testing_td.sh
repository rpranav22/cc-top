#!/bin/bash

NAME="DBP-P1_test"

CONFIG="dbpedia_td_p1.yaml"

# RUN = "testing_3_dt1"




sbatch --partition=mcml-dgx-a100-40x8 --time=00-18:00:00 --job-name=$NAME --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME-0 --error=/home/di39tih2/cc-top/error/$NAME-0 -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name testing_01_dt1 --test_set d_test_1
sleep 10s


sbatch --partition=mcml-dgx-a100-40x8 --time=00-18:00:00 --job-name=$NAME --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME-1 --error=/home/di39tih2/cc-top/error/$NAME-1 -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name testing_01_dt2 --test_set d_test_2
sleep 10s


sbatch --partition=mcml-dgx-a100-40x8 --time=00-18:00:00 --job-name=$NAME --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME-2 --error=/home/di39tih2/cc-top/error/$NAME-2 -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG --run_name testing_01_combined --test_set combined