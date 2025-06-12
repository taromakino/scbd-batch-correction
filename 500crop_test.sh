#!/bin/bash

python -m scbd_batch_correction.main \
    --dataset cellpainting2 \
    --data_dir /gstore/data/marioni_group/Carolina/CP2.0/CellCrops/cell_crops_500_test \
    --results_dir /gstore/data/marioni_group/Carolina/CP2.0/SCBD_analyses/outputs_test \
    --encoder_type resnet18 \
    --z_size 64 \
    --temperature 0.1 \
    --batch_size 32 \
    --y_per_batch 8 \
    --num_train_steps 1000 \
    --num_steps_per_val 100 \
    --limit_val_batches 50