"""
parallel_fusion_raytune.py

Description:
    Performs hyperparameter optimization for the parallel fusion (ConvNeXt + 
    Swin Transformer) model using the Ray Tune library. 

    Instead of training with a single set of predefined hyperparameters, this 
    script searches a defined 'search space' for the optimal combination of 
    learning rates and dropout rates. It leverages the ASHA (Asynchronous 
    Successive Halving Algorithm) scheduler to efficiently prune underperforming 
    trials, saving significant computation time.

    Each individual trial within the search still follows the two-phase 
    training logic (head-only training followed by full model fine-tuning). 
    After the search is complete, the script identifies the best trial, 
    loads its model checkpoint, and performs a final evaluation on the test set.

Usage:
    # For direct execution (e.g., on a local machine or for testing)
    python parallel_fusion_raytune.py

    # For submitting the job to the Slurm workload manager
    sbatch run_parallel_ray.sbatch

Inputs:
    - Dataset directory (set in script: base_data_dir): Must be structured 
      with 'train', 'val', and 'test' subdirectories.
    - Optional class list file (set in script: CLASS_LIST_FILE): A .txt file 
      to specify a subset of classes.
    - Hyperparameter search space (defined in script: search_space): A 
      dictionary that specifies the ranges for hyperparameters (e.g., 
      learning rates, dropout) to be explored by Ray Tune.

Outputs:
    - Ray Tune results directory (path set in script: output_dir): This is 
      the main output, containing a subdirectory for each hyperparameter 
      trial. Each trial folder includes its configuration, detailed logs, 
      and saved model checkpoints.
    - Final evaluation reports (for the single best model found):
        - A confusion matrix plot saved to the root of the output directory.
        - A summary of the best trial and its performance printed to the console.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path


# Import Ray Tune and the ASHA scheduler
import ray
from ray import tune, air
from ray.tune import Checkpoint,RunConfig, FailureConfig, CheckpointConfig

from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch


try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Seaborn not found. Confusion matrix will be printed as text.")

# Data Loading and Transforms 
# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Define image size (ConvNext Tiny and Swin Tiny use 224x224)
IMG_SIZE = 224

def get_data_transforms(img_size=IMG_SIZE):
    """
    Returns a dictionary of PyTorch transforms.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45), 
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def load_data(base_data_dir, batch_size=32, img_size=224, num_workers=1, num_classes_to_use=None, class_list_file=None):
    """
    Loads train, validation, and test data using ImageFolder and DataLoader.
    
    This version is corrected to prevent label mismatch between train/val/test splits.
    """
    data_transforms = get_data_transforms(img_size)
    

    print(f"Loading data from: {base_data_dir}")
    print(f"Using image size: {img_size}x{img_size}")
    print(f"DataLoader num_workers: {num_workers}")
    
    train_dir = Path(base_data_dir) / 'train'
    all_class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    
    class_names_to_use = []
    if class_list_file:
        class_file_path = Path(base_data_dir) / class_list_file
        print(f"Reading specific class list from: {class_file_path}")
        if not class_file_path.exists():
            raise FileNotFoundError(f"Specified class list file not found: {class_file_path}")
        
        with open(class_file_path, 'r') as f:
            requested_classes = {line.strip() for line in f if line.strip() and not line.strip().startswith('#')}
        
        class_names_to_use = [cls for cls in all_class_names if cls in requested_classes]
        missing_classes = requested_classes - set(class_names_to_use)
        if missing_classes:
            print(f"\nWarning: Requested classes not found and ignored: {', '.join(missing_classes)}\n")

    elif num_classes_to_use:
        print(f"Using the first {num_classes_to_use} available classes.")
        class_names_to_use = all_class_names[:num_classes_to_use]
    else:
        print("Using all available classes found in the directory.")
        class_names_to_use = all_class_names

    if not class_names_to_use:
        raise ValueError("No classes selected for training.")
        
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names_to_use)}
    num_classes = len(class_names_to_use)
    
    image_datasets = {}
    for phase in ['train', 'val', 'test']:
        phase_dir = Path(base_data_dir) / phase
        samples = []
        for class_name in class_names_to_use:
            class_dir = phase_dir / class_name
            if class_dir.is_dir():
                label = class_to_idx[class_name]
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif']:
                        samples.append((str(img_path), label))

        dataset = datasets.DatasetFolder(
            root=str(phase_dir.parent),
            loader=datasets.folder.default_loader,
            extensions=('.jpg', '.jpeg', '.png', '.tif'),
            transform=data_transforms[phase]
        )
        dataset.samples = samples
        dataset.classes = class_names_to_use
        dataset.class_to_idx = class_to_idx
        dataset.targets = [s[1] for s in samples]
        image_datasets[phase] = dataset

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    print(f"\nSuccessfully loaded and filtered for {num_classes} classes: {', '.join(class_names_to_use)}")
    print(f"Dataset sizes: Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}, Test: {dataset_sizes['test']}")

    return dataloaders, dataset_sizes, class_names_to_use, num_classes


# Model Definitions
class ParallelFusionModel(nn.Module):
    def __init__(self, num_classes, convnext_model_name='convnext_small', swin_model_name='swin_small_patch4_window7_224', pretrained=True, dropout_rate=0.5):
        super().__init__()
        self.convnext_model_name = convnext_model_name
        self.swin_model_name = swin_model_name

        self.convnext_model = timm.create_model(convnext_model_name, pretrained=pretrained, num_classes=0)
        convnext_feature_dim = self.convnext_model.num_features

        self.swin_model = timm.create_model(swin_model_name, pretrained=pretrained, num_classes=0)
        swin_feature_dim = self.swin_model.num_features

        self.fusion_dim = convnext_feature_dim + swin_feature_dim
        
        # Ray Tune uses the dropout_rate from the config 
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(1024, num_classes)
            )
        
    def forward(self, x, return_embeddings=False):
        features_convnext = self.convnext_model(x)
        features_swin = self.swin_model(x)
        fused_features = torch.cat((features_convnext, features_swin), dim=1)
        output = self.fusion_classifier(fused_features)
        if return_embeddings:
            return output, fused_features
        else:
            return output

def create_parallel_fusion_model(num_classes, convnext_model_name='convnext_tiny', swin_model_name='swin_tiny_patch4_window7_224', pretrained=True, dropout_rate=0.5):
    print(f"Creating Parallel Fusion model with ConvNext: {convnext_model_name}, Swin: {swin_model_name} for {num_classes} classes.")
    model = ParallelFusionModel(num_classes, convnext_model_name, swin_model_name, pretrained, dropout_rate)
    return model


# Plotting and Evaluation Functions 
def plot_training_history(history, phase_name="", output_dir="."):
    """Plots training and validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    title_prefix = f"{phase_name} " if phase_name else ""
    filename_prefix = phase_name.lower().replace(' ', '_') + "_" if phase_name else ""
    plot_save_path = os.path.join(output_dir, f"{filename_prefix}training_curves.png")
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title(f'{title_prefix}Loss vs. Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_title(f'{title_prefix}Accuracy vs. Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"Training curves saved to {plot_save_path}")
    plt.close(fig)

def evaluate_model(model, dataloader, device, class_names, criterion=None, output_dir="."):
    """
    Evaluates the model on a given dataloader and prints classification metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    print(f"\nEvaluating model on {dataset_size} test images...")

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if criterion:
        epoch_loss = running_loss / dataset_size
        print(f"Test Loss: {epoch_loss:.4f}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    cm_plot_save_path = os.path.join(output_dir, 'confusion_matrix_test_fusion.png')
    if SEABORN_AVAILABLE:
        plt.figure(figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.6)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (ConvNext-Swin Fusion)')
        plt.tight_layout()
        plt.savefig(cm_plot_save_path)
        print(f"Confusion matrix plot saved to {cm_plot_save_path}")
        plt.close()
    else:
        header = "True\\Pred | " + " | ".join(f"{name[:5]:<5}" for name in class_names)
        print(header)
        print("-" * len(header))
        for i, row in enumerate(cm):
            row_str = f"{class_names[i][:5]:<9} | " + " | ".join(f"{val:<5}" for val in row)
            print(row_str)

    return accuracy, precision_macro, recall_macro, f1_macro, cm


# Ray Tune 
# This function encapsulates the entire two-phase training logic for a single trial.
# Ray Tune will call this function multiple times with different hyperparameter configs.
def train_model_for_tune(config, data_ref=None):
    """
    Trainable function for Ray Tune.
    - config (dict): Hyperparameters for the trial.
    - data_ref (ObjectRef): Ray object store reference to the loaded data.
    """
    # SETUP 
    # Retrieve data from the Ray object store
  
    dataloaders, dataset_sizes, class_names, num_classes = data_ref
        
    # Model setup using hyperparameters from the config
    model = create_parallel_fusion_model(
        num_classes=num_classes,
        convnext_model_name='convnext_small',
        swin_model_name='swin_small_patch4_window7_224',
        pretrained=True,
        dropout_rate=config["dropout_rate"] # Use dropout from config
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # TRAINING PHASE 1: Fine-tune fusion classifier head 
    EPOCHS_PHASE1 = 20
    
    # Freeze backbones
    for param in model.convnext_model.parameters(): param.requires_grad = False
    for param in model.swin_model.parameters(): param.requires_grad = False
    for param in model.fusion_classifier.parameters(): param.requires_grad = True

    optimizer_phase1 = optim.AdamW(
        model.fusion_classifier.parameters(), 
        lr=config["lr_phase1_head"], # Use LR from config
        weight_decay=0.01
    )
    scheduler_phase1 = lr_scheduler.CosineAnnealingLR(optimizer_phase1, T_max=EPOCHS_PHASE1, eta_min=config["lr_phase1_head"]/100)

    for epoch in range(EPOCHS_PHASE1):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer_phase1.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer_phase1.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler_phase1:
                scheduler_phase1.step()
            
            # We only care about validation metrics for reporting to Tune
            if phase == 'val':
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
    # Save the state of the head-trained model to use for Phase 2
    phase1_best_wts = copy.deepcopy(model.state_dict())

    # TRAINING PHASE 2: Full model fine-tuning 
    EPOCHS_PHASE2 = 40
    model.load_state_dict(phase1_best_wts) # Load best weights from phase 1

    # Unfreeze all layers
    for param in model.parameters(): param.requires_grad = True

    optimizer_phase2 = optim.AdamW([
        {'params': model.convnext_model.parameters(), 'lr': config["lr_phase2_backbone"], 'name': 'ConvNext_Backbone'},
        {'params': model.swin_model.parameters(), 'lr': config["lr_phase2_backbone"], 'name': 'Swin_Backbone'},
        {'params': model.fusion_classifier.parameters(), 'lr': config["lr_phase2_head"], 'name': 'Fusion_Head'}
    ], weight_decay=0.01) # Use LRs from config
    
    scheduler_phase2 = lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=EPOCHS_PHASE2, eta_min=config["lr_phase2_backbone"]/10)

    for epoch in range(EPOCHS_PHASE2):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer_phase2.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer_phase2.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler_phase2:
                scheduler_phase2.step()
                
            if phase == 'val':
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
           
                # Report metrics and save a checkpoint for ASHA
                # The 'training_iteration' corresponds to the total number of epochs run
                checkpoint = None
                if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS_PHASE2: # Checkpoint every 5 epochs
                    os.makedirs("my_model", exist_ok=True)
                    torch.save(model.state_dict(), "my_model/checkpoint.pt")
                    checkpoint = Checkpoint.from_directory("my_model")
                
              
                metrics = {
                    "val_loss": epoch_loss,
                    "val_acc": epoch_acc.item()
                }
                
                # Pass the metrics dictionary. Ray handles the iteration count automatically.
                tune.report(metrics, checkpoint=checkpoint)

# Main Execution Block
if __name__ == '__main__':
    # Configuration 
    # Path to base data directory
    base_data_dir = 'path/to/directory'
    # Path to class list file
    CLASS_LIST_FILE = None # None to include all classes ; <Class list file name> If specifying classes to include in training and testing
    
    script_location_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to output directory
    output_dir = 'path/to/directory'
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Checkpoints and plots will be saved in: {output_dir}")

    CONVNEXT_MODEL_NAME = 'convnext_small'
    SWIN_MODEL_NAME = 'swin_small_patch4_window7_224' 
    NUM_WORKERS = 1
    NUM_CLASSES_TO_USE = None # or None to use all
    

    # Load data once and pass a reference to it into each trial.
    print("--- Loading Data (once for all trials) ---")
    dataloaders, dataset_sizes, class_names, num_classes = load_data(
        base_data_dir, batch_size=32, img_size=IMG_SIZE, num_workers=NUM_WORKERS,
        num_classes_to_use=NUM_CLASSES_TO_USE, class_list_file=CLASS_LIST_FILE
    )
    
    # Ray Tune Setup 
    # Put data into the Ray object store
    data_ref = ray.put((dataloaders, dataset_sizes, class_names, num_classes))

    # Define the hyperparameter search space
    search_space = {
        "lr_phase1_head": tune.loguniform(1e-4, 1e-2),
        "lr_phase2_head": tune.loguniform(5e-5, 5e-3),
        "lr_phase2_backbone": tune.loguniform(1e-5, 1e-4),
        "dropout_rate": tune.uniform(0.2, 0.6),
        # "batch_size": tune.choice([16, 32]), 
    }
    
    # Configure the ASHA scheduler
    asha_scheduler = ASHAScheduler(
        metric="val_acc",
        mode="max",
        max_t=60, # Max total epochs (20 for phase 1 + 40 for phase 2)
        grace_period=25, # Min epochs before a trial can be stopped
        reduction_factor=2 # Halve the number of trials every round
    )
   
    checkpoint_config = CheckpointConfig(
    num_to_keep=1,
    checkpoint_score_attribute="val_acc",
    checkpoint_score_order="max",
    )
    
    # Set up the Tuner
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model_for_tune, data_ref=data_ref),
            resources={"cpu": 2, "gpu": 1} # Resources per trial
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=20,  # Number of different hyperparameter sets to try
            scheduler=asha_scheduler,
            search_alg=HyperOptSearch(metric="val_acc", mode="max"),
        ),
        run_config=RunConfig(
            name="pollen_fusion_asha_tuning",
            storage_path=output_dir,
            checkpoint_config=checkpoint_config,  
            failure_config=FailureConfig(fail_fast=True),
        )
    )

    # Run Tuning 
    print("\n--- Starting Hyperparameter Tuning with Ray Tune and ASHA ---")
    results = tuner.fit()
    
    # Analyze Results and Final Evaluation 
    best_result = results.get_best_result(metric="val_acc", mode="max")
    
    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Best trial found: {best_result.path}")
    print(f"Best validation accuracy: {best_result.metrics['val_acc']:.4f}")
    print("Best hyperparameters found were: ", best_result.config)

    # Final Evaluation on Test Set using the Best Model 
    print("\n--- Starting Final Evaluation on Test Set with Best Model ---")
    
    # Create the model with the best hyperparameters
    best_config = best_result.config
    best_model = create_parallel_fusion_model(
        num_classes=num_classes,
        convnext_model_name=CONVNEXT_MODEL_NAME,
        swin_model_name=SWIN_MODEL_NAME,
        pretrained=False, 
        dropout_rate=best_config['dropout_rate']
    )
    
    # Load the best checkpoint from the best trial
    best_checkpoint = best_result.checkpoint
    with best_checkpoint.as_directory() as checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        best_model.load_state_dict(torch.load(checkpoint_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    # Evaluate
    evaluate_model(best_model, dataloaders['test'], device, class_names, criterion=nn.CrossEntropyLoss(), output_dir=output_dir)
    
    print("\nFULL SCRIPT WITH RAY TUNE COMPLETE.")