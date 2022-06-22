#!/bin/bash
for EXPID in {0..22}; do
    echo $EXPID;
    find mlruns/$EXPID -name 'confmat_*' -exec rm {} \;
    find mlruns/$EXPID -name '*.ckpt' -exec rm {} \;
    find mlruns/$EXPID -name '*final_model.pt' -exec rm {} \;
    find mlruns/$EXPID -name '*interm_model*' -exec rm {} \;
done