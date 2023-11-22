# import a libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance

# add a random seed for result repeatability
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# check for available devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}\n")

# link to the test dataset
dataset_test_root = r"Sentinel_2_dataset\test"

# retrieve a list of image paths from the test directory
all_filenames_test = []
for dirpath, dirnames, filenames in os.walk(dataset_test_root):
    for filename in filenames:
        if dirpath == dataset_test_root:
            continue
        all_filenames_test.append(os.path.join(dirpath, filename))
        
def select_samples(images):
    """
    Selects unique samples based on location and season from a list of image paths.

    Arguments:
    - images (list): List of image paths.

    Returns:
    - list(selected_images.values()): List of selected unique image paths.
    """
    selected_images = {}
    for img_path in images:
        parts = img_path.split(os.path.sep)
        location = parts[-2]
        season = parts[-1].split('_')[-2]
        key = f"{location}_{season}"
        if key not in selected_images:
            selected_images[key] = img_path
            
    return list(selected_images.values())

# select unique samples
all_filenames_test_prepared = select_samples(all_filenames_test)

print(f"There are {len(all_filenames_test_prepared)} instances for inference")

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

# load model weights
directory = "models"
model = SiameseNetwork()
model.load_state_dict(torch.load(os.path.join(directory, 'siamese_model.pth')))
model = model.to(device)

# create a dictionary with image file paths and their corresponding embeddings
embeddings = {}

with torch.inference_mode():
    for file in all_filenames_test_prepared:
        img = Image.open(file).convert('RGB')
        img_prepared = transform(img).unsqueeze(0)
        embedding = model.forward_once(img_prepared.to(device))
        embeddings[file] = embedding.cpu().detach().numpy()

def find_most_similar_embeddings(target_filename, embeddings_dict, top_n=5):
    """
    Finds the most similar embeddings to a target embedding from a dictionary of embeddings.

    Arguments:
    - target_filename (str): Filename of the target embedding.
    - embeddings_dict (dict): Dictionary containing filenames as keys and embeddings as values.
    - top_n (int): Number of most similar embeddings to retrieve. Defaults to 5.

    Returns:
    - most_similar (list): List of filenames for the most similar embeddings.
    """
    target_embedding = torch.tensor(embeddings_dict[target_filename])

    similarities = {}
    for filename, embedding in embeddings_dict.items():
        if filename != target_filename:
            embedding_tensor = torch.tensor(embedding)
            similarity_score = torch.cosine_similarity(target_embedding, embedding_tensor).item()
            similarities[filename] = similarity_score

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    most_similar = [filename for filename, _ in sorted_similarities[:top_n]]

    return most_similar

def calculate_map(filenames, embeddings_dict, top_n=2):
    """
    Calculates the Mean Average Precision (MAP) metric for a given set of filenames and embeddings.

    Arguments:
    - filenames (list): List of filenames.
    - embeddings_dict (dict): Dictionary containing filenames as keys and embeddings as values.
    - top_n (int): Number of most similar embeddings to consider. Defaults to 2.

    Returns:
    - mean_average_precision (float): Calculated mean average precision.
    """
    mean_average_precision = 0
    for file in filenames:
        true_positives = 0
        total_count = 0

        similar_files = find_most_similar_embeddings(file, embeddings_dict, top_n)
        for similar_file in similar_files:
            if os.path.basename(os.path.dirname(similar_file)) == os.path.basename(os.path.dirname(file)):
                true_positives += 1
            total_count += 1

        accuracy = true_positives / total_count if total_count > 0 else 0
        mean_average_precision += accuracy

    mean_average_precision /= len(filenames)
    return mean_average_precision

def calculate_top_n_accuracy(filenames, embeddings_dict, top_n=5):
    """
    Calculates the top-n accuracy for a given set of filenames and embeddings.

    Arguments:
    - filenames (list): List of filenames.
    - embeddings_dict (dict): Dictionary containing filenames as keys and embeddings as values.
    - top_n (int): Number of most similar embeddings to consider. Defaults to 5.

    Returns:
    - top_n_accuracy (float): Calculated top-n accuracy.
    """
    top_n_accuracy = 0
    for file in filenames:
        correct_prediction = 0

        similar_files = find_most_similar_embeddings(file, embeddings_dict, top_n)
        for i, similar_file in enumerate(similar_files, 1):
            if os.path.basename(os.path.dirname(similar_file)) == os.path.basename(os.path.dirname(file)):
                if i <= top_n:
                    correct_prediction = 1
                    break

        top_n_accuracy += correct_prediction

    top_n_accuracy /= len(filenames)
    return top_n_accuracy

# calculate metrics
map_1 = calculate_map(filenames=all_filenames_test_prepared, embeddings_dict=embeddings, top_n=1)
map_2 = calculate_map(filenames=all_filenames_test_prepared, embeddings_dict=embeddings, top_n=2)
map_3 = calculate_map(filenames=all_filenames_test_prepared, embeddings_dict=embeddings, top_n=3)


top_1 = calculate_top_n_accuracy(filenames=all_filenames_test_prepared, embeddings_dict=embeddings, top_n=1)
top_3 = calculate_top_n_accuracy(filenames=all_filenames_test_prepared, embeddings_dict=embeddings, top_n=3)
top_10 = calculate_top_n_accuracy(filenames=all_filenames_test_prepared, embeddings_dict=embeddings, top_n=10)
top_20 = calculate_top_n_accuracy(filenames=all_filenames_test_prepared, embeddings_dict=embeddings, top_n=20)

print(f"""
Mean Average Precision (mAP):
Top-1: {map_1 * 100:.2f}%
Top-2: {map_2 * 100:.2f}%
Top-3: {map_3 * 100:.2f}%

Top-N Accuracy:
Top-1: {top_1 * 100:.2f}%
Top-3: {top_3 * 100:.2f}%
Top-10: {top_10 * 100:.2f}%
Top-20: {top_20 * 100:.2f}%
""")
