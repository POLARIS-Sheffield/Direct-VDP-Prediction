#!/bin/bash
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import monai
import os
import glob
import numpy as np
import pandas as pd
import csv
import argparse
import pathlib
import time
import onnx
import shap

from pathlib import Path
from datetime import datetime

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    EnsureTyped,
    RandAffined,
    ScaleIntensityd,
    ConcatItemsd,
    ToNumpyd,
    Rotate90d,
    Rand3DElasticd,
)

from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, BasicUNet, AHNet, Regressor
from monai.networks.layers import Norm
from monai.metrics import MSEMetric, MAEMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, PersistentDataset, DataLoader
from monai.config import print_config
from monai.apps import download_and_extract
from monai.data.utils import list_data_collate, pad_list_data_collate
from monai.utils import first, set_determinism

############################### Set up ###########################################
parser = argparse.ArgumentParser()
parser.add_argument('-tdt','--train_data_root', help="train data spreadsheet root", type=Path, required=True)
parser.add_argument('-vdt','--val_data_root', help="validation data spreadsheet root", type=Path, required=True)
parser.add_argument('-sdt','--sest_data_root', help="test data spreadsheet root", type=Path, required=True)
args = parser.parse_args()

train_raw_dataset = str(args.train_data_root)
train_data_raw = pd.read_csv(train_raw_dataset, index_col="ID")

val_raw_dataset = str(args.val_data_root)
val_data_raw = pd.read_csv(val_raw_dataset, index_col="ID")

test_raw_dataset = str(args.sest_data_root)
test_data_raw = pd.read_csv(test_raw_dataset, index_col="ID")

with open("y_pred.csv", "a", newline="") as outcsv:
    writer = csv.DictWriter(outcsv, fieldnames = ["predicted VDP"])
    writer.writeheader()

with open("y_test.csv", "a", newline="") as outcsv:
    writer = csv.DictWriter(outcsv, fieldnames = ["predicted VDP"])
    writer.writeheader()
    
print_config()
##################################################################################

############################### Load Training Data ###############################
train_directory = '/VDP-prediction-git/HP_gas_vent_lung/dataset/training'
train_root_dir = tempfile.mkdtemp() if train_directory is None else train_directory
print(train_root_dir)

train_data_dir = os.path.join(train_root_dir, "image")
print(f"Training proton MRI directory: {train_data_dir}")

train_data_2_dir = os.path.join(train_root_dir, "image-vent")
print(f"Training hyperpolarised gas MRI directory: {train_data_2_dir}")

with open("train_data.csv", "a", newline="") as outcsv:
    writer = csv.DictWriter(outcsv, fieldnames = ["h1-image", "vent-image", "actual VDP"])
    writer.writeheader()

train_images = sorted(glob.glob(os.path.join(train_data_dir, "h1paired*.nii.gz")))
train_images_2 = sorted(glob.glob(os.path.join(train_data_2_dir, "image*.nii.gz")))
train_labels = train_data_raw["VDP"]

train_data_dicts = [{"image": image_name, "image-vent": image_2_name, "labels": label_name} for image_name, image_2_name, label_name in zip(train_images, train_images_2, train_labels)]

train_files = train_data_dicts

print(f"Number of Training cases: {len(train_files)}")

for item in train_files:
    print(item, file=open("train_data.csv", "a"))
##################################################################################

############################# Load Validation Data ###############################
val_directory = '/VDP-prediction-git/HP_gas_vent_lung/dataset/validation'
val_root_dir = tempfile.mkdtemp() if val_directory is None else val_directory
print(val_root_dir)

val_data_dir = os.path.join(val_root_dir, "image")
print(f"Validation proton MRI directory: {val_data_dir}")

val_data_2_dir = os.path.join(val_root_dir, "image-vent")
print(f"Validation hyperpolarised gas MRI directory: {val_data_2_dir}")

with open("validation_data.csv", "a", newline="") as outcsv:
    writer = csv.DictWriter(outcsv, fieldnames = ["h1-image", "vent-image", "actual VDP"])
    writer.writeheader()

val_images = sorted(glob.glob(os.path.join(val_data_dir, "h1paired*.nii.gz")))
val_images_2 = sorted(glob.glob(os.path.join(val_data_2_dir, "image*.nii.gz")))
val_labels = val_data_raw["VDP"]

val_data_dicts = [{"image": image_name, "image-vent": image_2_name, "labels": label_name} for image_name, image_2_name, label_name in zip(val_images, val_images_2, val_labels)]

val_files = val_data_dicts

print(f"Number of Validation cases: {len(val_files)}")

for item in val_files:
    print(item, file=open("validation_data.csv", "a"))
##################################################################################

############################# Load Testing Data ###############################
test_directory = '/VDP-prediction-git/HP_gas_vent_lung/dataset/testing'
test_root_dir = tempfile.mkdtemp() if test_directory is None else test_directory
print(test_root_dir)

test_data_dir = os.path.join(test_root_dir, "image")
print(f"Testing proton MRI directory: {test_data_dir}")

test_data_2_dir = os.path.join(test_root_dir, "image-vent")
print(f"Testing hyperpolarised gas MRI directory: {test_data_2_dir}")

with open("testing_data.csv", "a", newline="") as outcsv:
    writer = csv.DictWriter(outcsv, fieldnames = ["h1-image", "vent-image", "actual VDP"])
    writer.writeheader()

test_images = sorted(glob.glob(os.path.join(test_data_dir, "h1paired*.nii.gz")))
test_images_2 = sorted(glob.glob(os.path.join(test_data_2_dir, "image*.nii.gz")))
test_labels = test_data_raw["VDP"]

test_data_dicts = [{"image": image_name, "image-vent": image_2_name, "labels": label_name} for image_name, image_2_name, label_name in zip(test_images, test_images_2, test_labels)]

test_files = test_data_dicts

print(f"Number of Testing cases: {len(test_files)}")

for item in test_files:
    print(item, file=open("testing_data.csv", "a"))
##################################################################################

#################################### Transforms ##################################
set_determinism(seed=1)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "image-vent"]),
        ToNumpyd(keys=["labels"]),
        EnsureChannelFirstd(keys=["image", "image-vent", "labels"], channel_dim=-1),
        Rotate90d(keys=["image", "image-vent"], k=-1),
        ScaleIntensityd(keys=["image", "image-vent"], channel_wise=True, allow_missing_keys=True),
        EnsureTyped(keys=["image", "image-vent", "labels"]),
        RandAffined(
            keys=["image", "image-vent"],
            mode=('bilinear', 'nearest'),
            prob=0.1,
            rotate_range=(0, np.pi/15, np.pi/15),
            scale_range=(0, 0.15, 0.15),
            lazy=True),
        Rand3DElasticd(
            keys=["image", "image-vent"],
            mode=("bilinear", "nearest"),
            prob=0.5,
            sigma_range=(5, 8),
            magnitude_range=(50, 200),
            rotate_range=(0, np.pi/15, np.pi/15),
            scale_range=(0, 0.15, 0.15)),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image"], meta_keys="filename"),
        LoadImaged(keys=["image-vent"]),
        ToNumpyd(keys=["labels"]),
        EnsureChannelFirstd(keys=["image", "image-vent", "labels"], channel_dim=-1),
        Rotate90d(keys=["image", "image-vent"], k=-1),
        ScaleIntensityd(keys=["image", "image-vent"], channel_wise=True, allow_missing_keys=True),
        EnsureTyped(keys=["image", "image-vent", "labels"]),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=["image"], meta_keys="filename"),
        LoadImaged(keys=["image-vent"]),
        ToNumpyd(keys=["labels"]),
        EnsureChannelFirstd(keys=["image", "image-vent", "labels"], channel_dim=-1),
        Rotate90d(keys=["image", "image-vent"], k=-1),
        ScaleIntensityd(keys=["image", "image-vent"], channel_wise=True, allow_missing_keys=True),
        EnsureTyped(keys=["image", "image-vent", "labels"]),
    ]
)
##################################################################################

############################ Check + visualise data ##############################
check_ds = Dataset(data=val_files, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)

image_1, image_2 = (check_data["image"][0], check_data["image-vent"][0])
plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Proton MRI")
plt.imshow(image_1[10, :, :], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Hyperpolarised gas MRI")
plt.imshow(image_2[10, :, :], cmap="gray")
plt.savefig('/VDP-prediction-git/example_scan_pair.png')
plt.close()
##################################################################################

#################################### DataLoader ##################################
train_ds = CacheDataset(data=train_files, transform=train_transforms, num_workers=8)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, collate_fn=pad_list_data_collate)

val_ds = CacheDataset(data=val_files, transform=val_transforms, num_workers=8)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8, collate_fn=pad_list_data_collate)

test_ds = CacheDataset(data=test_files, transform=test_transforms, num_workers=8)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8, collate_fn=pad_list_data_collate)
##################################################################################

################################## Define Model ##################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Regressor(
        in_shape=(2, 256, 256, 20),
        out_shape=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        kernel_size=3,
        num_res_units=3,
        act='prelu',
        norm='INSTANCE',
        dropout=0.15,
        bias=True).to(device)

print("#model_params:", np.sum([len(p.flatten()) for p in model.parameters()]))

loss_function = torch.nn.SmoothL1Loss(beta=0.55)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=700)
mse_metric = MAEMetric(reduction="mean")

max_epochs = 2
val_interval = 1
##################################################################################

################################## Train loop ####################################
best_metric = 500
best_metric_epoch = -1
epoch_loss_values = []
val_loss_values = []
metric_loss_values = []
metric_loss_values_test = []
metric_values = []
metric_values_test = []
tic = time.time()
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        result = ConcatItemsd(keys=["image", "image-vent"], name="img")(batch_data)
        inputs, labels = (result["img"], batch_data["labels"])
        optimizer.zero_grad()
        inputs = inputs.unsqueeze(0)
        inputs = inputs.permute(0, 1, 3, 4, 2)
        labels = labels.unsqueeze(1)
        output = model(inputs.to(device))
        loss = loss_function(output, labels.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}, Train_loss: {epoch_loss/step:.4f}", "\r", end="")
        
    scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} Average train loss: {epoch_loss:.4f} time elapsed: {(time.time()-tic)/60:.2f} mins")
    
    input_names = ["Combined HP gas MRI and Proton MRI"]
    output_names = ["VDP Prediction"]
    torch.onnx.export(model, inputs, "model.onnx", input_names=input_names, output_names=output_names)
##################################################################################

############################### Validation loop ##################################
    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_loss = 0
        step = 0
        metric_loss = 0
        with torch.no_grad():
            for val_data in val_loader:
                step += 1
                result = ConcatItemsd(keys=["image", "image-vent"], name="img")(val_data)
                inputs, labels = (result["img"], val_data["labels"])
                inputs = inputs.unsqueeze(0)
                inputs = inputs.permute(0, 1, 3, 4, 2)
                labels = labels.unsqueeze(1)
                outputs = model(inputs.to(device))
                with open("y_pred.csv", "a", newline="") as outcsv:
                    writer = csv.writer(outcsv)
                    writer.writerow(outputs)
                vloss = loss_function(outputs, labels.to(device))
                val_loss += vloss.item()
                print(f"Validation_loss: {vloss:.4f}")
                
                metricloss = mse_metric(y_pred=outputs, y=labels)
                metricloss.int()
                metric_loss += metricloss.item()

            scheduler.step()
            val_loss /= step
            val_loss_values.append(val_loss)
            print(f"Epoch {epoch + 1} Average validation loss: {val_loss:.4f}")

            metric_loss /= step
            metric_loss_values.append(metric_loss)
            print(f"Epoch {epoch + 1} Average MAE value: {metric_loss:.4f}")
            
            if metric_loss < best_metric:
                best_metric = metric_loss
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(test_root_dir, "best_metric_model.pth"))
                print("Saved new best metric model")
                
            print(
                f"Best Average MAE: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

print(f"Training completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
##################################################################################

###################### Plot training / validation curves #########################
plt.figure("train", (12, 6))
plt.subplot(1, 3, 1)
plt.title("Average Training Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Epoch")
plt.plot(x, y)

plt.subplot(1, 3, 2)
plt.title("Average Validation Loss")
x = [val_interval * (i + 1) for i in range(len(val_loss_values))]
y = val_loss_values
plt.xlabel("Epoch")
plt.plot(x, y)

plt.subplot(1, 3, 3)
plt.title("Validation Average MSE")
x = [val_interval * (i + 1) for i in range(len(metric_loss_values))]
y = metric_loss_values
plt.xlabel("Epoch")
plt.plot(x, y)

plt.savefig('/VDP-prediction-git/train-validation-curves.png')
plt.close()
##################################################################################

############################### Testing loop ##################################
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

model.load_state_dict(torch.load(os.path.join(test_root_dir, "best_metric_model.pth")))
model.eval()
step = 0
metric_loss_test = 0
forward_passes = 10
with torch.no_grad():
    for i in range(forward_passes):
        enable_dropout(model)
        for test_data in test_loader:
            step += 1
            result_test = ConcatItemsd(keys=["image", "image-vent"], name="img")(test_data)
            inputs_test, labels_test = (result_test["img"], test_data["labels"])
            inputs_test = inputs_test.unsqueeze(0)
            inputs_test = inputs_test.permute(0, 1, 3, 4, 2)
            labels_test = labels_test.unsqueeze(1)
            outputs_test = model(inputs_test.to(device))
            with open("y_test.csv", "a", newline="") as outcsv:
                writer = csv.writer(outcsv)
                writer.writerow(outputs_test)

            metriclosstest = mse_metric(y_pred=outputs_test, y=labels_test)
            metriclosstest.int()
            metric_loss_test += metriclosstest.item()
        
    metric_loss_test /= step
    metric_loss_values_test.append(metric_loss_test)
    print(f"Best Epoch {best_metric_epoch} Average Testing MAE value: {metric_loss_test:.4f}")
##################################################################################

################################# Clean up csv ###################################
text = open("y_pred.csv", "r")
text = ''.join([i for i in text]).replace("metatensor([", "")
x = open("y_pred.csv","w")
x.writelines(text)
x.close()

text4 = open("y_pred.csv", "r")
text4 = ''.join([i for i in text4]).replace("])", "")
x = open("y_pred.csv","w")
x.writelines(text4)
x.close()

text4 = open("validation_data.csv", "r")
text4 = ''.join([i for i in text4]).replace("'labels': ", "")
x = open("validation_data.csv","w")
x.writelines(text4)
x.close()

text4 = open("validation_data.csv", "r")
text4 = ''.join([i for i in text4]).replace("}", "")
x = open("validation_data.csv","w")
x.writelines(text4)
x.close()

text = open("y_test.csv", "r")
text = ''.join([i for i in text]).replace("metatensor([", "")
x = open("y_test.csv","w")
x.writelines(text)
x.close()

text4 = open("y_test.csv", "r")
text4 = ''.join([i for i in text4]).replace("])", "")
x = open("y_test.csv","w")
x.writelines(text4)
x.close()
##################################################################################
