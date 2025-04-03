#!/bin/bash

python test_brier_denoised.py -tdt /VDP-prediction-git/HP_gas_vent_lung/dataset/training/labels/training_labels.csv -vdt /VDP-prediction-git/HP_gas_vent_lung/dataset/validation/labels/validation_labels.csv -sdt /VDP-prediction-git/HP_gas_vent_lung/dataset/testing/labels/testing_labels.csv
