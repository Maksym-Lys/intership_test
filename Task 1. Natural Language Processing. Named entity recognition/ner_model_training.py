# import libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForTokenClassification
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# define a hyperparameters
BATCH_SIZE = 32
MAX_LENGTH = 128
EPOCHS = 20
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2

# check for available devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}\n")

# add a random seed for result repeatability
g = torch.Generator().manual_seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# read dataset CSV file
ner_df = pd.read_csv("Dataset.csv")
tokens = ner_df["Words"]
labels = ner_df["Labels"]

# split instances into train, dev and test
train_df, inference_df = train_test_split(ner_df, test_size=TEST_SIZE, random_state=101)
dev_df, _ = train_test_split(inference_df, test_size=0.5, random_state=101)

print(f"""
Created DataFrames

Train dataset length: {len(train_df)} instances.
Dev dataset length:   {len(dev_df)} instances.
""")

# create label map for encoding entities
label_map = {"O": 1, 
             "B-mountain": 2, "I-mountain": 2,
             "B-country": 3, "I-country": 3,
             "B-location": 4, "I-location": 4,
             "B-scientist": 5, "I-scientist": 5,
             "B-astronomicalobject": 6,  "I-astronomicalobject": 6,
             "B-organisation": 7, "I-organisation": 7,
             "B-award": 8, "I-award": 8,
             "B-misc": 9, "I-misc": 9,
             "B-academicjournal": 10, "I-academicjournal": 10,
             "B-university": 11, "I-university": 11,
             "B-person": 12, "I-person": 12,
             "B-chemicalcompound": 13, "I-chemicalcompound": 13,
             "B-protein": 14, "I-protein": 14,
             "B-event": 15, "I-event": 15,
             "B-enzyme": 16, "I-enzyme": 16,
             "B-discipline": 17, "I-discipline": 17,
             "B-theory": 18, "I-theory": 18,
             "B-chemicalelement":19, "I-chemicalelement":19}

#  define BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def return_text(x):
    """
    Joins a list of strings into a single string with spaces in between each element.

    Arguments:
    - x (list of str): List of strings to be concatenated.

    Returns:
    - text (str): A single string containing all elements of the list separated by spaces.
    """
    return ' '.join(x)

def tokenize_and_align_labels(row):
    """
    Tokenizes words and aligns corresponding labels to each token.

    Arguments:
    - row (dict): Dictionary containing 'Words' and 'Labels' keys with strings as values.

    Returns:
    - tokens (list): List of tokens obtained from the words after tokenization.
    - token_labels (list): List of labels aligned with each token.
    """
    tokens = []
    token_labels = []
    words = row["Words"].split()
    labels = row["Labels"].split()

    for word, label in zip(words, labels):
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        token_labels.extend([label] * len(word_tokens))

    return tokens, token_labels

def create_dataloader_from_df(df, generator, batch_size, max_length, shuffle=True):
    """
    Creates a PyTorch DataLoader from a DataFrame by tokenizing inputs and labels.

    Arguments:
    - df (pandas.DataFrame): Input DataFrame containing columns 'Words' and 'Labels'.
    - generator (torch.Generator): PyTorch random number generator.
    - batch_size (int): Batch size for the DataLoader.
    - max_length (int): Maximum length of token sequences.
    - shuffle (bool): Flag to shuffle data in DataLoader (default=True).

    Returns:
    - dataloader (torch.utils.data.DataLoader): PyTorch DataLoader for the tokenized inputs and labels.
    """
    
    # create input_tensor and attention_mask
    tokenized_data = df.apply(tokenize_and_align_labels, axis=1)
    df["Tokens"] = tokenized_data.apply(lambda x: x[0])
    df["Token_Labels"] = tokenized_data.apply(lambda x: x[1])
    df["Clean_Words"] = df["Tokens"].apply(return_text)
    df["Clean_Labels"] = df["Token_Labels"].apply(return_text)
    tokenized_inputs = tokenizer.batch_encode_plus(
        df["Words"].tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
        add_special_tokens=False)
    input_tensor = tokenized_inputs["input_ids"]
    attention_masks = tokenized_inputs["attention_mask"]
    
    # create label tensor
    tokenized_labels = []
    for sentence in df["Clean_Labels"]:
        row_of_labels = []
        for label in sentence.split():
            idx = label_map[label]
            row_of_labels.append(idx)
        tokenized_labels.append(row_of_labels)
    padded_labels = []
    for sequence in tokenized_labels:
        if len(sequence) >= max_length:
            padded_sequence = sequence[:max_length]  # Truncate if longer than max_length
        else:
            padded_sequence = sequence + [0] * (max_length - len(sequence))  # Pad with zeros
        padded_labels.append(padded_sequence)
    output_tensor = torch.tensor(padded_labels)
    
    print(f"Created dataloader with elements: {input_tensor.shape}, {attention_masks.shape}, {output_tensor.shape}")
    dataset = TensorDataset(input_tensor, attention_masks ,output_tensor)
    dataloader = DataLoader(dataset, generator=generator, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

# create dataloader for training and testing
train_dataloader = create_dataloader_from_df(train_df, 
                                             batch_size=BATCH_SIZE, 
                                             max_length=MAX_LENGTH,
                                             generator=g)

dev_dataloader = create_dataloader_from_df(dev_df, 
                                           batch_size=BATCH_SIZE, 
                                           max_length=MAX_LENGTH, 
                                           shuffle=False,
                                           generator=g)

def train_step(model, dataloader, optimizer, scheduler, device):
    """
    Perform one training step.

    Arguments:
    - model (torch.nn.Module): The model to train.
    - dataloader (torch.utils.data.DataLoader): DataLoader for training data.
    - optimizer (torch.optim.Optimizer): Optimizer used for training.
    - scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    - device (torch.device): Device to perform computations.

    Returns:
    - avg_loss (float): Average loss over the training data.
    - class_2_accuracy (float): Accuracy for class 2 tokens in the training data.
    """
    model.train()
    total_loss = 0.0
    total_correct_class_2 = 0
    total_class_2 = 0

    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=2)

        batch_class_2_mask = (labels == 2)  # Mask for class 2 tokens
        batch_correct_class_2 = ((predicted_labels == labels) & batch_class_2_mask).sum().item()
        batch_class_2 = batch_class_2_mask.sum().item()

        total_correct_class_2 += batch_correct_class_2
        total_class_2 += batch_class_2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    class_2_accuracy = total_correct_class_2 / total_class_2 if total_class_2 > 0 else 0.0
    return avg_loss, class_2_accuracy


def eval_step(model, dataloader, device):
    """
    Perform one evaluation step.

    Arguments:
    - model (torch.nn.Module): The model to evaluate.
    - dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
    - device (torch.device): Device to perform computations.

    Returns:
    - avg_loss (float): Average loss over the evaluation data.
    - class_2_accuracy (float): Accuracy for class 2 tokens in the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_correct_class_2 = 0
    total_class_2 = 0

    with torch.inference_mode():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=2)

            batch_class_2_mask = (labels == 2)
            batch_correct_class_2 = ((predicted_labels == labels) & batch_class_2_mask).sum().item()
            batch_class_2 = batch_class_2_mask.sum().item()

            total_correct_class_2 += batch_correct_class_2
            total_class_2 += batch_class_2

    avg_loss = total_loss / len(dataloader)
    class_2_accuracy = total_correct_class_2 / total_class_2 if total_class_2 > 0 else 0.0
    return avg_loss, class_2_accuracy


def train(model,
          train_dataloader, 
          dev_dataloader, 
          optimizer,
          scheduler,
          epochs,
          device):
    """
    Train the model using the provided data.

    Arguments:
    - model (torch.nn.Module): The model to train.
    - train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
    - dev_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
    - optimizer (torch.optim.Optimizer): Optimizer used for training.
    - scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    - epochs (int): Number of training epochs.
    - device (torch.device): Device to perform computations.

    Returns:
    - results (dict): Dictionary containing training and validation metrics.
    """
    results = {"train_loss": [],
    "train_acc (class 'mountain')": [],
    "val_loss": [],
    "val_acc (class 'mountain')": []
    }

    for epoch in range(epochs):

        train_loss, class_2_accuracy = train_step(model, 
                                                  train_dataloader, 
                                                  optimizer, 
                                                  scheduler, 
                                                  device)

        val_loss, val_class_2_accuracy = eval_step(model, 
                                                   dev_dataloader, 
                                                   device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc (class 'mountain'): {class_2_accuracy:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc (class 'mountain'): {val_class_2_accuracy:.4f}"
            )

        results["train_loss"].append(train_loss)
        results["train_acc (class 'mountain')"].append(class_2_accuracy)
        results["val_loss"].append(val_loss)
        results["val_acc (class 'mountain')"].append(val_class_2_accuracy)

    return results

# model initialization and training
model_name = "bert-base-uncased"
model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(label_map))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * EPOCHS)

results = train(model=model,
                train_dataloader=train_dataloader, 
                dev_dataloader=dev_dataloader, 
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=EPOCHS,
                device=device)

# save model
model_path = "models/"
model.save_pretrained(model_path)
print(f"The model weights are saved at the path: {model_path}")
