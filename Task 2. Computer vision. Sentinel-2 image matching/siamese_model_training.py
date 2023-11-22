# import a libraries
import re
import os
import numpy as np

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

from collections import defaultdict

# define a hyperparameters
BATCH_SIZE = 16
MARGIN = 1
EPOCHS = 5
LEARNING_RATE = 0.0002

# add a random seed for result repeatability
g = torch.Generator().manual_seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# check for available devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}\n")

# links to the train and dev datasets
dataset_train_root = r"Sentinel_2_dataset\train"
dataset_dev_root = r"Sentinel_2_dataset\dev"

# retrieve a list of image paths from the dataset directories
all_filenames_train = []
for dirpath, dirnames, filenames in os.walk(dataset_train_root):
    for filename in filenames:
        if dirpath == dataset_train_root:
            continue
        all_filenames_train.append(os.path.join(dirpath, filename))
        
all_filenames_dev = []
for dirpath, dirnames, filenames in os.walk(dataset_dev_root):
    for filename in filenames:
        if dirpath == dataset_dev_root:
            continue
        all_filenames_dev.append(os.path.join(dirpath, filename))
        
def group_by_class_and_season(image_paths):
    """
    Groups image paths by class and season.

    Arguments:
    - image_paths (list): List of file paths to images.

    Returns:
    - classes (defaultdict): Dictionary containing image paths grouped by class and season.
    """
    classes = defaultdict(lambda: defaultdict(list))
    
    for path in image_paths:
        class_name = os.path.basename(os.path.dirname(path))
        season = os.path.splitext(os.path.basename(path))[0].split('_')[-2]
        classes[class_name][season].append(path)
    
    return classes

def generate_triplets(image_paths, seed=42):
    """
    Generates triplets of image paths for use in triplet loss.

    Arguments:
    - image_paths (list): List of file paths to images.
    - seed (int): Seed for random number generation. Defaults to 42.

    Returns:
    - triplets (list): List of image triplets, each consisting of an anchor, positive, and negative image path.
    """
    random.seed(seed)
    classes = group_by_class_and_season(image_paths)
    triplets = []
    
    for class_name, class_data in classes.items():
        for season, paths in class_data.items():
            for anchor_path in paths:
                same_class = [img for img in paths if img != anchor_path]
                
                # Positive: Same class, different season
                positive_seasons = [s for s in class_data.keys() if s != season]
                positive_candidates = [img for s in positive_seasons for img in class_data.get(s, [])]
                positive_candidates = [img for img in positive_candidates if img != anchor_path]  # Remove anchor from positives
                positive = random.choice(positive_candidates) if positive_candidates else random.choice(paths)
                
                # Negative: Different class, same season
                anchor_class_num = int(''.join(filter(str.isdigit, class_name)))
                anchor_class_hundreds = (anchor_class_num // 100) * 100
                
                different_class = []
                for cls, seasons in classes.items():
                    if anchor_class_hundreds <= int(''.join(filter(str.isdigit, cls))) < anchor_class_hundreds + 100 and cls != class_name:
                        different_class.extend(seasons.get(season, []))
                
                different_class = [img for img in different_class if img != anchor_path]  # Remove anchor from negatives
                negative = random.choice(different_class) if different_class else random.choice(paths)
                
                if (anchor_class_num != negative) and (anchor_class_num % 100 != 0):
                    triplets.append((paths[0], positive, negative))  # Using the first image of the class as anchor_path

    return triplets

# create the train and test triplets
triplets_train = generate_triplets(all_filenames_train)
triplets_dev = generate_triplets(all_filenames_dev)

print(f"Created train triplets with {len(triplets_train)} instances")
print(f"Created dev triplets with {len(triplets_dev)} instances\n")

class TripletDataset(Dataset):
    """
    Dataset class for handling triplets of images.

    Arguments:
    - triplets (list): List of triplets, each containing paths to anchor, positive, and negative images.
    - transform (callable, optional): Optional transformations to be applied to the images. Defaults to None.
    """
    def __init__(self, triplets, transform=None):
        """
        Initializes the TripletDataset instance.

        - triplets: List of triplets, each containing paths to anchor, positive, and negative images.
        - transform: Optional transformations to be applied to the images. Defaults to None.
        """
        self.triplets = triplets
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of triplets in the dataset.
        """
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        Retrieves a specific triplet (anchor, positive, negative) from the dataset.

        - idx: Index to retrieve the triplet.

        Returns:
        - anchor_img: Image data for the anchor image.
        - positive_img: Image data for the positive image.
        - negative_img: Image data for the negative image.
        """
        anchor_path, positive_path, negative_path = self.triplets[idx]

        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img
    
class BrightnessEnhancement(object):
    """
    Custom transformation class to enhance the brightness of an image.

    Arguments:
    - factor (float): Enhancement factor for brightness adjustment.
    """
    def __init__(self, factor):
         """
        Initializes the BrightnessEnhancement transformation.

        - factor: Enhancement factor for brightness adjustment.
        """
        self.factor = factor

    def __call__(self, img):
        """
        Performs the brightness enhancement on the input image.

        - img: Input image to be enhanced.

        Returns:
        - enhanced_img: Image with adjusted brightness.
        """
        enhancer = ImageEnhance.Brightness(img)
        enhanced_img = enhancer.enhance(self.factor)
        return enhanced_img

# Define the transformations
transform = transforms.Compose([
    BrightnessEnhancement(factor=3),
    transforms.ToTensor()
])

# Creating datasets and dataloders
train_dataset = TripletDataset(triplets_train, transform=transform)
dev_dataset = TripletDataset(triplets_dev, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, generator=g)
dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, generator=g)

print(f"Created train dataloader: {train_dataloader}")
print(f"Created dev dataloader: {dev_dataloader}\n")

class SiameseNetwork(nn.Module):
    """
    Siamese Neural Network architecture for triplet loss.

    This network takes in triplets of images (anchor, positive, negative),
    processes them through shared convolutional layers, and produces embeddings.

    Architecture:
    - Base Convolutional Neural Network with customizable layers.
    - Fully connected layers for embedding.

    Methods:
    - __init__: Initializes the SiameseNetwork.
    - forward_once: Forward pass through the network for a single image.
    - forward: Forward pass for anchor, positive, and negative samples.
    """
    def __init__(self):
        """
        Initializes the SiameseNetwork architecture.

        Defines a base convolutional neural network and fully connected layers for embedding.
        """
        super().__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )

    def forward_once(self, x):
        """
        Forward pass through the base network and fully connected layers for a single image.

        - x: Input image tensor.

        Returns:
        - Embedding tensor for the input image.
        """
        x = self.conv_base(x)
        x = x.view(x.size()[0], -1)  
        x = self.fc(x)
        return x

    def forward(self, anchor, positive, negative):
        """
        Forward pass for anchor, positive, and negative samples.

        - anchor: Tensor for the anchor image.
        - positive: Tensor for the positive image.
        - negative: Tensor for the negative image.

        Returns:
        - Embeddings for the anchor, positive, and negative images.
        """
        anchor_emb = self.forward_once(anchor)
        positive_emb = self.forward_once(positive)
        negative_emb = self.forward_once(negative)
        return anchor_emb, positive_emb, negative_emb


# Triplet Loss function
class TripletLoss(nn.Module):
    """
    Triplet Loss function for training Siamese Networks.

    Arguments:
    - margin (float): Margin value for the loss calculation. Defaults to 1.0.
    """
    def __init__(self, margin=1.0):
        """
        Initializes the TripletLoss function.

        - margin: Margin value for the loss calculation.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Forward pass of the TripletLoss function.

        - anchor: Embeddings for the anchor images.
        - positive: Embeddings for the positive images.
        - negative: Embeddings for the negative images.

        Returns:
        - mean_loss: Mean loss computed across the batch.
        """
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)

        losses = torch.relu(distance_positive - distance_negative + self.margin)

        mean_loss = torch.mean(losses)
        return mean_loss

class ModelCheckpoint:
    """
    Model checkpointing utility to save the best model based on validation loss.

    Arguments:
    - save_path (str): Path to save the model.
    - model_name (str): Name to use for saving the model file.
    - save_best_only (bool): Flag to save only the best model. Defaults to True.
    - verbose (int): Verbosity level for printing messages. Defaults to 1.
    """
    def __init__(self, save_path, model_name, save_best_only, verbose):
        """
        Initializes the ModelCheckpoint.

        - save_path: Path to save the model.
        - model_name: Name to use for saving the model file.
        - save_best_only: Flag to save only the best model. Defaults to True.
        - verbose: Verbosity level for printing messages. Defaults to 1.
        """
        self.save_path = save_path
        self.model_name = model_name
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0

        os.makedirs(save_path, exist_ok=True)
        
    def __call__(self, val_loss, model):
        """
        Method to call the ModelCheckpoint during training.

        - val_loss: Validation loss obtained during training.
        - model: The model to be saved.

        Actions:
        - Saves the model if the validation loss improves.
        - Optionally prints messages based on verbosity and saving conditions.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), os.path.join(self.save_path, self.model_name))
            if self.verbose > 0:
                print(f"Saved model with validation loss: {val_loss:.4f}.")  
        else:
            if not self.save_best_only:
                torch.save(model.state_dict(), os.path.join(self.save_path, self.model_name))
                if self.verbose > 0:
                    print(f"Saved model with validation loss: {val_loss:.4f}.")  
                    
            if self.verbose > 0:
                print(f"Model's validation loss: {self.best_loss:.4f} didn't improve.")
    
class EarlyStopping:
    """
    Early Stopping utility to stop training when the validation loss stops improving.

    Arguments:
    - patience (int): Number of epochs to wait for improvement before stopping. Defaults to 5.
    - min_delta (float): Minimum change in the monitored quantity to qualify as improvement. Defaults to 0.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        Initializes the EarlyStopping.

        - patience: Number of epochs to wait for improvement before stopping.
        - min_delta: Minimum change in the monitored quantity to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        """
        Method to call the EarlyStopping during training.

        - val_loss: Validation loss obtained during training.

        Returns:
        - early_stop (bool): Flag indicating whether to stop training or not.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def train_step(model, dataloader, optimizer, loss_fn, device):
    """
    Performs a single training step.

    - model: The neural network model being trained.
    - dataloader: Data loader providing training data.
    - optimizer: Optimization algorithm (e.g., SGD, Adam).
    - loss_fn: Loss function (e.g., TripletLoss) for calculating the loss.
    - device: Device to move the data (e.g., CPU, GPU).

    Returns:
    - running_loss / len(dataloader): Average training loss per batch.
    """
    model.train()
    running_loss = 0.0

    for i, data in enumerate(dataloader):
        anchor, positive, negative = data
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        optimizer.zero_grad()
        output_anchor, output_positive, output_negative = model(anchor, positive, negative)
        loss = loss_fn(output_anchor, output_positive, output_negative)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(dataloader)


def eval_step(model, dataloader, loss_fn, device):
    """
    Evaluates the model on validation or test data.

    - model: The neural network model being evaluated.
    - dataloader: Data loader providing validation/test data.
    - loss_fn: Loss function (e.g., TripletLoss) for calculating the loss.
    - device: Device to move the data (e.g., CPU, GPU).

    Returns:
    - total_loss / len(dataloader): Average evaluation loss per batch.
    """
    model.eval()
    total_loss = 0.0

    with torch.inference_mode():
        for i, data in enumerate(dataloader):
            anchor, positive, negative = data
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            output_anchor, output_positive, output_negative = model(anchor, positive, negative)
            loss = loss_fn(output_anchor, output_positive, output_negative)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)


def train(model, train_dataloader, dev_dataloader, optimizer, loss_fn, epochs, model_checkpoint, early_stopping, device):
    """
    Performs the training loop over multiple epochs.

    - model: The neural network model being trained.
    - train_dataloader: Data loader for training data.
    - dev_dataloader: Data loader for validation data.
    - optimizer: Optimization algorithm (e.g., SGD, Adam).
    - loss_fn: Loss function (e.g., TripletLoss) for calculating the loss.
    - epochs: Number of epochs for training.
    - model_checkpoint: Utility for saving the best model based on validation loss.
    - early_stopping: Utility to stop training if validation loss stops improving.
    - device: Device to move the data (e.g., CPU, GPU).

    Returns:
    - results: Dictionary containing training and validation losses per epoch.
    """
    results = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        train_loss = train_step(model, train_dataloader, optimizer, loss_fn, device)
        val_loss = eval_step(model, dev_dataloader, loss_fn, device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f}"
        )

        if early_stopping is not None:
            if early_stopping(val_loss=val_loss):
                print("Early stopping triggered")
                break
        if model_checkpoint is not None:
            model_checkpoint(val_loss, model)
            
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

    return results

# model initialization and training
model = SiameseNetwork().to(device)
triplet_loss = TripletLoss(margin=MARGIN)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
early_stopping = EarlyStopping(patience=1)
checkpoint = ModelCheckpoint(save_path='models/', model_name='siamese_model.pth', save_best_only=True, verbose=1)

print("Start training")
results = train(
    model=model,
    train_dataloader=train_dataloader,
    dev_dataloader=dev_dataloader, 
    optimizer=optimizer,
    loss_fn=triplet_loss,
    model_checkpoint=checkpoint,
    early_stopping=early_stopping,
    epochs=EPOCHS,
    
    device=device)
print("Finish training\n")
