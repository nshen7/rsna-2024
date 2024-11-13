import utils
import numpy as np
from engine import train_one_epoch, evaluate
from sklearn.metrics import precision_score, recall_score, f1_score

import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

# Util functions
with open(os.path.join(SRC_DIR, 'data.py')) as file:
    exec(file.read())
with open(os.path.join(SRC_DIR, 'models.py')) as file:
    exec(file.read())


# SEVERITY_LABELS = {
#     "Normal/Mild": 0,
#     "Moderate": 1,
#     "Severe": 2
# }

num_classes = 3

def model_pipeline(config, model, model_dir, train_df, val_df, device, datasetter=RSNACroppedImageDataset, limit=None):
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=config):
        
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, val_loader, criterion, optimizer, lr_scheduler = make(config, datasetter, model, train_df, val_df, limit)
        
        # and use them to train the model
        train_and_validate(model, model_dir, train_loader, val_loader, criterion, optimizer, lr_scheduler, config, device)

        # # and test its final performance
        # validate(model, val_loader, criterion, device)

    return

def make(config, datasetter, model, train_df, val_df, limit):
    
    # Make training set
    dataset = datasetter(train_df, limit = limit)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    # Make validation set
    dataset_val = datasetter(val_df, limit = limit)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config['batch_size'],
        shuffle=True
    )

    # Use a weighted cross entropy loss, where 
    #    weight=1 for normal/mild,
    #    weight=2 for moderate,
    #    weight=4 for severe.
    class_weights = torch.tensor([1.0, 2.0, 4.0])  # Adjust weights as needed
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Move model and critirion to the right device
    criterion.to(device)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params,
        lr=config['lr'],
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_step_size'],
        gamma=config['lr_gamma']
    )
    
    return model, train_loader, val_loader, criterion, optimizer, lr_scheduler


def train_and_validate(model, model_dir, train_loader, val_loader, criterion, optimizer, lr_scheduler, config, device):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(train_loader) * config['num_epochs']
    example_ct = 0  # number of examples seen
    for epoch in tqdm(range(config['num_epochs'])):
        
        all_labels = []
        all_preds = []
        batch_ct = 0

        for _, (images, labels) in enumerate(train_loader):

            loss, outputs = train_batch(images, labels, model, optimizer, criterion, device)
                
            # Record true labels and predicted labels for monitoring accuracy
            _, preds = outputs.max(1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
    
            example_ct += len(images)
            batch_ct += 1

            # Report metrics every 20th batch
            if ((batch_ct + 1) % 20) == 0:

                # Calculate accuracy metrics
                accuracy, precision, recall, f1 = accuracy_metrics(all_labels, all_preds)
                
                # Log the metrics
                train_log(loss, accuracy, precision, recall, f1, example_ct, batch_ct, epoch)

                # Reset accumulators for the next interval
                all_labels = []
                all_preds = []
        
        # update the learning rate
        lr_scheduler.step()
        
        # Save model after every epoch
        dirname = f'{model_dir}/epoch_{epoch}'
        os.makedirs(dirname, exist_ok=True,)
        fname = f'{dirname}/model_dict.pt'
        torch.save(model.state_dict(), fname)
        
        
        # Print vaidation metrics
        model.eval()
        val_labels = []
        val_preds = []
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Collect predictions and labels
                _, preds = torch.max(outputs, 1)
                val_labels.extend(labels.tolist())
                val_preds.extend(preds.tolist())


        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)

        # Calculate accuracy, precision, and recall for the validation set
        val_accuracy, val_precision, val_recall, val_f1 = accuracy_metrics(val_labels, val_preds)

        # Print validation metrics
        print(f"Validation Metrics:")
        print(f"  Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {val_accuracy * 100:.2f}%")
        for cls in range(num_classes):
            print(f"  Class {cls}: Precision: {val_precision[cls]:.4f}, Recall: {val_recall[cls]:.4f}, F1-score: {val_f1[cls]:.4f}")

        wandb.log({"test_loss": avg_val_loss, "test_accuracy": val_accuracy, 
                   "test_precision": val_precision, "test_recall": val_recall, "test_f1": val_f1})


# ---------------- util functions --------------

def train_log(loss, accuracy, precision, recall, f1, example_ct, batch_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy, 
               "precision": precision, "recall": recall, "f1": f1}, step=example_ct)
    
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
    print(f"Batch {batch_ct + 1} Metrics:")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    for cls in range(num_classes):
        print(f"  Class {cls}: Precision: {precision[cls]:.4f}, Recall: {recall[cls]:.4f}, F1-score: {f1[cls]:.4f}")
    

def train_batch(images, labels, model, optimizer, criterion, device):
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss, outputs


def accuracy_metrics(labels, preds):
    # Convert to tensors for metric calculations
    labels_tensor = torch.tensor(labels)
    preds_tensor = torch.tensor(preds)
    
    # Compute the metrics
    accuracy = (preds_tensor == labels_tensor).sum().item() / len(labels_tensor)
    precision = precision_score(labels, preds, average=None, labels=range(num_classes), zero_division=np.nan)
    recall = recall_score(labels, preds, average=None, labels=range(num_classes), zero_division=np.nan)
    f1 = f1_score(labels, preds, average=None, labels=range(num_classes), zero_division=np.nan)
    
    return accuracy, precision, recall, f1