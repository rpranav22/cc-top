#!/bin/bash
# bash script to reproduce table 5 on the 
# Dynamic Topic Discovery

for DSET in "agnews" "dbpedia" "trec"; do 
    for REP in 0 1 2 3 4; do
        python run_text.py --config configs/"${DSET}_clustering.yaml" --run_name $RUN-$REP
        python run_text.py --config configs/"${DSET}_constrained.yaml" --loss MCL --run_name $RUN-$REP
        python run_text.py --config configs/"${DSET}_constrained.yaml" --loss KCL --run_name $RUN-$REP
        python run_text.py --config configs/"${DSET}_supervised.yaml" --run_name $RUN-$REP
    done
done    