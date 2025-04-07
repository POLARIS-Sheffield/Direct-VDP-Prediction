# Direct-VDP-Prediction

Code and trained model to directly predict VDP from 1H-MRI and 129Xe-MRI scans.

## Image Paths

The user must provide paths to folders containing the hyperpolarized gas MRI and proton MRI scans for the training, validation and testing sets.

```
train_directory = '/VDP-prediction-git/HP_gas_vent_lung/dataset/training'
val_directory = '/VDP-prediction-git/HP_gas_vent_lung/dataset/validation'
test_directory = '/VDP-prediction-git/HP_gas_vent_lung/dataset/testing'
```

## VDP label Paths

VDP labels can be provided in the accompanying bash script.

## Running the code

The following code can be used to run the VDP prediction script.

```
python test_brier_denoised.py -tdt /VDP-prediction-git/HP_gas_vent_lung/dataset/training/labels/training_labels.csv -vdt /VDP-prediction-git/HP_gas_vent_lung/dataset/validation/labels/validation_labels.csv -sdt /VDP-prediction-git/HP_gas_vent_lung/dataset/testing/labels/testing_labels.csv
```

