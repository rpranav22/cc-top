# CCTop

## run things

* For local GPU: from `experiments` run `python run_text.py --config configs/dbpedia_constraint.yaml`
* For SLURM run: from `experiments` run `bash example_run_slurm.sh` which uses slurm lingo

## work with cluster

* `squeue` to check traffic on cluster
* `srun --partition=mcml-dgx-a100-40x8 --gres=gpu:1 --qos=mcml --pty bash` to attach one GPU to your terminal (so you can debug, use it in interactive mode)
* `scancel <job id>` to cancel one job with given id, `scancel {startID..stopID}` to kill a series of slurm jobs
* error/ out messages are stored in `error/out` folders in cc-top repository