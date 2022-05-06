#!/usr/bin/env python
# coding: utf-8

from IPython.terminal.interactiveshell import TerminalInteractiveShell
import random
import glob
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import matplotlib
import matplotlib.font_manager as fm
from typing import Dict, List, Tuple
import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models as models
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import timm
from timm.data.auto_augment import rand_augment_transform
import madgrad


### Initialize shell for execute shell commands from python file

shell = TerminalInteractiveShell.instance()
shell.system('nvidia-smi')
# shell.system('wget  -O chula-food-50.zip https://piclab.ai/classes/cv2021/Chula-food-50.zip')
# shell.system('unzip -qo chula-food-50.zip')


### Config & Device

config = {
    "model_name": 'convnext_base_384_in22ft1k',
    "dropout": 0.2,
    "batch_size": 16,
    "image_size": (384, 384),
    "seed": 42,
    "image_path": os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'full_clean_images_improved'),
    "min_sample_per_class": 60,
    "val_length": 15,
    "test_length": 15,
    "num_workers": 2,
    "num_epochs": 15,
    "weight_filepath": os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_weight.pt'),
    "result_filepath": os.path.join(os.path.dirname(os.path.realpath(__file__)), 'result.csv'),
}

random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Dataset/Dataloader useful class & functions

class CustomImageFolder(ImageFolder):
    def __init__(self, excluded_classes=None, **kwargs):
        self.excluded_classes = set(excluded_classes) if excluded_classes else set()
        super().__init__(**kwargs)

    # Override the find_classes method to allow excluding some classes that the size is too small
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_dir() and entry.name not in self.excluded_classes
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def get_df_dataset(dataset, print_idx_to_class=False):
    df_dataset = pd.DataFrame(
        dataset.imgs,
        columns=["image_path", "class_idx"],
    )
    df_class_to_idx = pd.DataFrame(
        dataset.class_to_idx.items(),
        columns=["class_name", "class_idx"],
    )
    df_dataset = df_dataset.merge(df_class_to_idx, on="class_idx")
    if(print_idx_to_class):
        print("\nclass_index to class_name mapping :") 
        print({v: k for k, v in dataset.class_to_idx.items()})
    return df_dataset


def find_low_size_classes(image_path, threshold):
    dataset = ImageFolder(root=image_path)
    df_dataset = get_df_dataset(dataset)
    df_class_count = (
        df_dataset.groupby("class_name")
        .count()
        .rename(columns={"image_path": "sample_size"})
    )
    low_size_classes = set(df_class_count[df_class_count.sample_size < threshold].index)
    return low_size_classes


def get_train_val_test_indices(dataset, random_state=None):
    df_dataset = get_df_dataset(dataset, True)

    val_length = config["val_length"]
    test_length = config["test_length"]
    val_test_length = val_length + test_length

    # Create a list of indices we will use as validation and test set
    val_test_indices = (
        df_dataset.groupby("class_idx")
        .sample(n=val_test_length, random_state=random_state)
        .index
    )
    df_train = df_dataset.drop(index=val_test_indices)
    df_val_test = df_dataset.loc[val_test_indices]
    test_indices = (
        df_dataset.groupby("class_idx")
        .sample(n=test_length, random_state=random_state)
        .index
    )
    df_val = df_val_test.drop(index=test_indices)
    df_test = df_val_test.loc[test_indices]
    train_indices = df_train.index
    val_indices = df_val.index
    test_indices = df_test.index

    return train_indices, val_indices, test_indices


def get_weighted_random_sampler(train_dataset, train_indices):
    subset_targets = np.asarray(train_dataset.targets)[train_indices]
    label_weights = 1 / np.bincount(subset_targets)
    weights = label_weights[subset_targets]
    sampler = WeightedRandomSampler(
        weights,
        len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    return sampler


### Transform (have augment for train but no augment for validation and test)

train_transform = transforms.Compose(
    [
        transforms.Resize(size=config["image_size"]),
        rand_augment_transform(
            config_str="rand-m9-mstd0.5",
            hparams={},
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(size=config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

### Dataset => split to train, val, test => Dataloader 

invalid_classes = find_low_size_classes(
    config["image_path"], config["min_sample_per_class"]
)

train_dataset = CustomImageFolder(
    root=config["image_path"],
    transform=train_transform,
    excluded_classes=invalid_classes,
)
test_dataset = CustomImageFolder(
    root=config["image_path"],
    transform=test_transform,
    excluded_classes=invalid_classes,
)

train_indices, val_indices, test_indices = get_train_val_test_indices(
    train_dataset, random_state=config["seed"]
)

train_split = Subset(train_dataset, train_indices)
val_split = Subset(test_dataset, val_indices)
test_split = Subset(test_dataset, test_indices)

# Use WeightedRandomSampler to tackle the class imbalance problem
sampler = get_weighted_random_sampler(train_dataset, train_indices)

train_loader = DataLoader(
    train_split,
    batch_size=config["batch_size"],
    sampler=sampler,
    num_workers=config["num_workers"],
    pin_memory=True,
    generator=torch.Generator().manual_seed(config["seed"]),
)
val_loader = DataLoader(
    val_split,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=True,
)
test_loader = DataLoader(
    test_split,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=True,
)


### Model

class foodNet(nn.Module):
  def __init__(self, num_classes):
    super(foodNet, self).__init__()
    ### Layers goes here ###
    self.pretrained_model = timm.create_model(config["model_name"], pretrained=True, drop_rate=config["dropout"])
    self.pretrained_model.head.fc = nn.Linear(self.pretrained_model.head.fc.in_features, num_classes)

  def forward(self, input):
    ### Connections goes here ###
    x = self.pretrained_model(input)
    return x
    

print(f'\nNumber of class: {len(train_dataset.classes)}')

model = foodNet(len(train_dataset.classes))

for parameter in model.pretrained_model.parameters():
    parameter.requires_grad_(False)

for parameter in model.pretrained_model.head.parameters():
    parameter.requires_grad_(True)

model.to(device)


### Loss function / Optimizer / Scheduler

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = madgrad.MADGRAD(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
# optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()))
scheduler = optim.lr_scheduler.CosineAnnealingLR(
  optimizer=optimizer, 
  T_max=config["num_epochs"],
  eta_min=1e-7,
  verbose=True
)
# scheduler = optim.lr_scheduler.CyclicLR(
#     optimizer=optimizer,
#     base_lr=1e-6,
#     max_lr=5e-5,
#     step_size_up= 4 * len(train_split) // BATCH_SIZE,
#     cycle_momentum=False,
# )



### Training model

scaler = amp.GradScaler()
### Train, Validate and Test helper functions ###
# def testModel(testDatasetLoader, net):
#   net.eval()
#   numClass = len(testDatasetLoader.dataset.dataset.classes)
#   correctImagesPerClass = np.zeros(numClass)
#   totalImagesPerClass = np.zeros(numClass)
#   testingProgressbar = tqdm(enumerate(testDatasetLoader), total=len(testDatasetLoader), ncols=100)
#   with torch.no_grad():
#     for batchIdx, batchData in testingProgressbar:
#       images, labels = batchData[0].to(device), batchData[1].to(device)
      
#       outputs = net(images)
#       _, predicted = torch.max(outputs, 1)

#       labels = labels.cpu().detach().numpy()
#       predicted = predicted.cpu().detach().numpy()

#       correctImagesPerClass += np.bincount(labels[labels==predicted], minlength=numClass)
#       totalImagesPerClass += np.bincount(labels, minlength=numClass)

#   accuracyPerClass = correctImagesPerClass / totalImagesPerClass
#   accumulateAccuracy = round((correctImagesPerClass.sum().item()/totalImagesPerClass.sum().item())*100, 4)
#   return accuracyPerClass, accumulateAccuracy

def testModel(testDatasetLoader, net):
  net.eval()
  numClass = len(testDatasetLoader.dataset.dataset.classes)
  top1CorrectImagesPerClass = np.zeros(numClass)
  top3CorrectImagesPerClass = np.zeros(numClass)
  totalImagesPerClass = np.zeros(numClass)
  testingProgressbar = tqdm(enumerate(testDatasetLoader), total=len(testDatasetLoader), ncols=100)
  with torch.no_grad():
    for batchIdx, batchData in testingProgressbar:
      images, labels = batchData[0].to(device), batchData[1].to(device)
      
      outputs = net(images)
      _, top3Predicted = torch.topk(outputs, 3)

      labels = labels.cpu().detach().numpy()
      top3Predicted = top3Predicted.cpu().detach().numpy()
      rank1Predicted = top3Predicted[:,0]
      rank2Predicted = top3Predicted[:,1]
      rank3Predicted = top3Predicted[:,2]

      top1CorrectImagesPerClass += np.bincount(labels[labels==rank1Predicted], minlength=numClass)
      top3CorrectImagesPerClass += np.bincount(labels[(labels==rank1Predicted) | (labels==rank2Predicted) | (labels==rank3Predicted)], minlength=numClass)
      totalImagesPerClass += np.bincount(labels, minlength=numClass)

  top1AccuracyPerClass = top1CorrectImagesPerClass / totalImagesPerClass
  top1AccumulateAccuracy = round((top1CorrectImagesPerClass.sum().item()/totalImagesPerClass.sum().item())*100, 4)
  top3AccuracyPerClass = top3CorrectImagesPerClass / totalImagesPerClass
  top3AccumulateAccuracy = round((top3CorrectImagesPerClass.sum().item()/totalImagesPerClass.sum().item())*100, 4)
  return top1AccuracyPerClass, top1AccumulateAccuracy, top3AccuracyPerClass, top3AccumulateAccuracy


def validateModel(valDatasetLoader, net):
  net.eval()
  sumLoss = 0.0
  sampleCount = 0
  correctImages = 0
  totalImages = 0
  testingProgressbar = tqdm(enumerate(valDatasetLoader), total=len(valDatasetLoader), ncols=100)
  with torch.no_grad():
    for batchIdx, batchData in testingProgressbar:
      images, labels = batchData[0].to(device), batchData[1].to(device)
      sampleCount += len(images)
      
      outputs = net(images)
      loss = criterion(outputs, labels)
      sumLoss += loss.item() * len(images)

      _, predicted = torch.max(outputs, 1)
      correctImages += (predicted == labels).sum().item()
      totalImages += labels.size(0)

      accumulateAccuracy = round((correctImages/totalImages)*100,4)
      accumulateLoss = round(sumLoss/sampleCount,4)
      testingProgressbar.set_description("Validation accuracy: {} loss: {}".format(accumulateAccuracy, accumulateLoss ) )
  
  print(f'Val Accuracy: {accumulateAccuracy}, Val Loss: {accumulateLoss}')
  return accumulateAccuracy, accumulateLoss


def trainAndValidateModel(trainDatasetLoader, valDatasetLoader, net, optimizer,scheduler, criterion, trainEpoch):
  bestAccuracy = 0
  correctImages = 0
  totalImages = 0
  for currentEpoch in tqdm(range(trainEpoch), desc='Overall Training Progress:', ncols=100):
    sumLoss = 0.0
    sampleCount = 0
    net.train()
    print('Epoch',str(currentEpoch+1),'/',str(trainEpoch))
    trainingProgressbar = tqdm(enumerate(trainDatasetLoader), total=len(trainDatasetLoader), ncols=100)
    for batchIdx, batchData in trainingProgressbar:
      images, labels = batchData[0].to(device), batchData[1].to(device)
      sampleCount += len(images)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      with amp.autocast():
        outputs = net(images)
        loss = criterion(outputs, labels)
        sumLoss += loss.item() * len(images)
    
      _, predicted = torch.max(outputs, 1)
      correctImages += (predicted == labels).sum().item()
      totalImages += labels.size(0)
    
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      
      accumulateAccuracy = round((correctImages/totalImages)*100,4)
      accumulateLoss = round(sumLoss/sampleCount,4)
      trainingProgressbar.set_description("Training accuracy: {} loss: {}".format(accumulateAccuracy, accumulateLoss ) )
    
    print(f'Train Accuracy: {accumulateAccuracy}, Train Loss: {accumulateLoss}')
    valAccuracy, valLoss = validateModel(valDatasetLoader, net)
    scheduler.step()  #not using accumulateLoss or valLoss because we use cosineAnnealingLR to step LR once per epoch
    print('='*10)
    
    if valAccuracy > bestAccuracy:
      bestAccuracy = valAccuracy
      bestNet = net
      torch.save(bestNet.state_dict(), config["weight_filepath"])

  return bestAccuracy, bestNet


# In[ ]:

bestAccuracy, bestNet = trainAndValidateModel(
    train_loader,
    val_loader,
    model,
    optimizer,
    scheduler,
    criterion,
    config["num_epochs"],
)

# model.load_state_dict(torch.load(config["weight_filepath"], map_location=device))

# for parameter in model.pretrained_model.stages[2:].parameters():
#     parameter.requires_grad_(True)

# for name, param in model.named_parameters():
#   if param.requires_grad:
#     print(name)

# torch.cuda.empty_cache() 

# bestAccuracy, bestNet = trainAndValidateModel(
    # train_loader,
    # val_loader,
    # model,
    # optimizer,
    # scheduler,
    # criterion,
    # config["num_epochs"],
# )


### Predict test dataset

model.load_state_dict(torch.load(config["weight_filepath"], map_location=device))
top1AccuracyPerClass, top1Accuracy, top3AccuracyPerClass, top3Accuracy = testModel(test_loader, model)


### Export predicted result

result_df = pd.DataFrame({
    'class_name': test_dataset.classes,
    'top1_accuracy': top1AccuracyPerClass,
    'top3_accuracy': top3AccuracyPerClass,
    'num_training_data': np.bincount(np.asarray(train_dataset.targets)[train_split.indices], minlength=len(train_dataset.classes)),
})
result_df.to_csv(config["result_filepath"], index=False)

print('======================================Result======================================')
print(f'Number of class: {len(test_dataset.classes)}')
print(f'Val Accuracy: {bestAccuracy}')
print(f'Top-1 Test Accuracy: {top1Accuracy}')
print(f'Top-3 Test Accuracy: {top3Accuracy}')
