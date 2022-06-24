#!/bin/bash
# bash script to repeat best model over folds

# Specify the config file you want to tune
CONFIG="yahoo_constrained.yaml"

# give the child a name
NAME='YHOq-test'

IT=0

# sbatch sends slurm run command to free gpu, until --gres everything is slurm options that tell slurm 
# which/ how many gpu to run/ where your home directory is etc. lalal
sbatch --partition=mcml-dgx-a100-40x8 --time=00-18:00:00 --job-name=$NAME-$IT --qos=mcml --output=/home/di39tih2/cc-top/out/$NAME-$IT --error=/home/di39tih2/cc-top/error/$NAME-$IT -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG

# # 2nd run
# CONFIG="dbpedia_supervised.yaml"
# # give the child a name
# NAME='DBP'
# IT=1
# sbatch --partition=mcml-dgx-a100-40x8 --time=00-18:00:00 --job-name=$NAME-$IT --qos=mcml --output=/home/di39tih2/out/$NAME-$IT --error=/home/di39tih2/cc-top/error/$NAME-$IT -D /home/di39tih2/cc-top/experiments/ --gres=gpu:1 /home/di39tih2/cc-top/experiments/run_text.py --config /home/di39tih2/cc-top/experiments/configs/$CONFIG



# If you want to be smart, you can loop over different values 

# TRASH for repeated runs/ hparams

# for FOLD in 0 1 2 3 4; do
#     sbatch --partition=mcml-dgx-a100-40x8 --job-name=$NAME-$IT --qos=mcml --output=/home/di39tih2/out/$NAME-$IT --error=/home/di39tih2/error/$NAME-$IT -D /home/di39tih2/sscc/experiments/ --gres=gpu:1 /home/di39tih2/sscc/experiments/run.py --config /home/di39tih2/sscc/experiments/configs/$CONFIG --run_name $NAME-$FOLD
#     if (( $IT == 0)); then
#         sleep 60s
#     fi
#     ((IT+=1))
#     sleep 5s
# done