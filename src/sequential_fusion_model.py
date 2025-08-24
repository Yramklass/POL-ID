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
from sklearn.metrics import classification_report
import pandas as pd

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

# Define image size for CoAtNet (coatnet_1_rw_224 expects 224x224)
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
            # transforms.TrivialAugmentWide(), # Auto-augmentation- reduced accuracy
            transforms.RandomRotation(45), 
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size + 32), 
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        'test': transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
    }
    return data_transforms

def load_data(base_data_dir, batch_size=32, img_size=IMG_SIZE, num_workers=1):
    """
    Loads train, validation, and test data using ImageFolder and DataLoader.
    """
    data_transforms = get_data_transforms(img_size)

    print(f"Loading data from: {base_data_dir}")
    print(f"Using image size: {img_size}x{img_size}")
    print(f"DataLoader num_workers: {num_workers}") 

    image_datasets = {}
    for x in ['train', 'val', 'test']:
        data_path = os.path.join(base_data_dir, x)
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"Data directory for '{x}' not found at {data_path}")
        image_datasets[x] = datasets.ImageFolder(data_path, data_transforms[x])

    # Ensure test loader is not shuffled for consistent evaluation
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    
    if not image_datasets['train'].classes:
        raise ValueError("No classes found in the training dataset. Check your data directory structure.")
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    print(f"Found {num_classes} classes: {', '.join(class_names)}")
    print(f"Dataset sizes: Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}, Test: {dataset_sizes['test']}")

    return dataloaders, dataset_sizes, class_names, num_classes


# Model Creation Function 

def create_coatnet_model(num_classes, model_name='coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k', pretrained=True):
    """
    Creates a CoAtNet model using timm.
    """
    print(f"Creating CoAtNet model: {model_name} with {num_classes} classes.")
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model

# Training Function 
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, 
                num_epochs=25, device=None, checkpoint_name='best_model.pth'):
    """
    Trains the model and saves the best weights.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nStarting training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                if scheduler: 
                    scheduler.step() 

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if optimizer.param_groups: # Check if optimizer has param_groups
                 current_lr = optimizer.param_groups[0]['lr']
                 print(f'{phase.capitalize()} Current LR: {current_lr:.7f}')


            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), checkpoint_name)
                print(f'Best val Acc: {best_acc:.4f} -> Model saved to {checkpoint_name}')
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# Plotting Function 
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

# Evaluation Function 
def evaluate_model(model, dataloader, device, class_names, criterion=None, output_dir="."):
    """
    Evaluates the model and provides overall metrics, per-class metrics,
    a confusion matrix, a CSV report, and a performance bar chart.
    Includes error handling for the per-class metric generation.
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

    # --- OVERALL METRICS ---
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")

    # --- NEW: ROBUST PER-CLASS METRICS CALCULATION AND SAVING ---
    try:
        print("\nPer-Class Metrics:")
        # Use classification_report, output_dict=True makes it easy to process
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
        
        # Convert to a Pandas DataFrame for saving and display
        df_report = pd.DataFrame(report).transpose()
        print(df_report)

        # Save the report to a CSV file
        csv_save_path = os.path.join(output_dir, 'per_class_metrics.csv')
        df_report.to_csv(csv_save_path)
        print(f"\nPer-class metrics saved to {csv_save_path}")

        # Plot and save per-class F1-scores
        if SEABORN_AVAILABLE:
            per_class_f1 = {name: report[name]['f1-score'] for name in class_names}
            
            plt.figure(figsize=(max(10, len(class_names) * 0.5), 6))
            sns.barplot(x=list(per_class_f1.keys()), y=list(per_class_f1.values()), palette="viridis")
            plt.xlabel('Pollen Class')
            plt.ylabel('F1-Score')
            plt.title('Per-Class F1-Scores')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.tight_layout()

            plot_save_path = os.path.join(output_dir, 'per_class_f1_scores.png')
            plt.savefig(plot_save_path)
            print(f"Per-class F1-score plot saved to {plot_save_path}")
            plt.close()

    except Exception as e:
        # Catch any error during the report generation and print a warning
        print(f"\nWarning: Could not generate or save per-class metrics.")
        print(f"   Error: {e}")
        print("   Skipping per-class report and plot, but continuing with confusion matrix...")

    # --- CONFUSION MATRIX (existing code runs regardless of the above try...except) ---
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    
    cm_plot_save_path = os.path.join(output_dir, 'confusion_matrix_test.png')
    if SEABORN_AVAILABLE:
        plt.figure(figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.6)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(cm_plot_save_path)
        print(f"Confusion matrix plot saved to {cm_plot_save_path}")
        plt.close()
    else:
        # Fallback text-based confusion matrix
        header = "True\\Pred | " + " | ".join(f"{name[:5]:<5}" for name in class_names)
        print(header)
        print("-" * len(header))
        for i, row in enumerate(cm):
            row_str = f"{class_names[i][:5]:<9} | " + " | ".join(f"{val:<5}" for val in row)
            print(row_str)
    
    return accuracy, precision_macro, recall_macro, f1_macro, cm


# Main Execution Block 
if __name__ == '__main__':
    # Configuration 
    base_data_dir = "/scratch/rmkyas002/processed_crops" 
    
    if not os.path.isdir(base_data_dir):
        print(f"ERROR: HPC Data directory not found at '{base_data_dir}'")
        exit()
    
    script_location_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_location_dir, "training_outputs")
    os.makedirs(output_dir, exist_ok=True) 
    print(f"Checkpoints and plots will be saved in: {output_dir}")


    MODEL_NAME = 'coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k' # Imagenet 12k version
    BATCH_SIZE = 16 
    NUM_WORKERS = 1 
    print(f"Using NUM_WORKERS = {NUM_WORKERS} for DataLoaders.")


    # Phase 1: Train only the head
    LR_PHASE1 = 1e-3
    EPOCHS_PHASE1 = 25 
    CHECKPOINT_PHASE1 = os.path.join(output_dir, 'pollen_coatnet_phase1_head_best.pth')

    # Phase 2: Fine-tune the entire model
    LR_BACKBONE_PHASE2 = 1e-4 
    LR_HEAD_PHASE2 = 5e-4    
    EPOCHS_PHASE2 = 50 
    CHECKPOINT_PHASE2 = os.path.join(output_dir, 'pollen_coatnet_phase2_full_best.pth')

    # Load Data 
    try:
        dataloaders, dataset_sizes, class_names, num_classes = load_data(
            base_data_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=NUM_WORKERS
        )
    except Exception as e:
        print(f"Error during data loading: {e}")
        exit()

    # Create Model 
    pollen_model = create_coatnet_model(num_classes=num_classes, model_name=MODEL_NAME)
    
    # Setup Device & CUDA Diagnostics
    print(f"\n--- Device Setup & CUDA Diagnostics ---")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device index: {torch.cuda.current_device()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
             print(f"Device {i} name: {torch.cuda.get_device_name(i)}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        # print(f"CUDNN version: {torch.backends.cudnn.version()}") # Requires cudnn to be visible
    else:
        print("CUDA is NOT available. PyTorch will use CPU.")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pollen_model.to(device)
    print(f"Model '{MODEL_NAME}' loaded on device: {device}")

    # Unweighted loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Calculate class weights for weighted loss function 
    # print("\nCalculating class weights to handle imbalance...")
    # class_counts = np.bincount(dataloaders['train'].dataset.targets)
    # class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    # class_weights = class_weights / class_weights.sum() * num_classes # Normalize
    # class_weights = class_weights.to(device)

    # print(f"Class weights calculated and moved to {device}.")


    # criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    # PHASE 1: Train the classifier head 
    print(f"\n--- Starting Training Phase 1: Fine-tuning classifier head ---")
    print(f"Epochs: {EPOCHS_PHASE1}, LR: {LR_PHASE1}")

    for param in pollen_model.parameters():
        param.requires_grad = False

    params_to_train_phase1_list = []
    if hasattr(pollen_model, 'head') and hasattr(pollen_model.head, 'fc'):
        print("Identified classifier head at 'model.head.fc'. Unfreezing for Phase 1.")
        for param in pollen_model.head.fc.parameters():
            param.requires_grad = True
        params_to_train_phase1_list = list(pollen_model.head.fc.parameters())
    else:
        print("ERROR: Could not find 'model.head.fc'. Please inspect model structure and adjust.")
        exit()
    
    if not params_to_train_phase1_list: 
         print("ERROR: No parameters selected for training in Phase 1. Check head unfreezing logic.")
         exit()
    
    optimizer_phase1 = optim.AdamW(params_to_train_phase1_list, lr=LR_PHASE1, weight_decay=0.01)
    scheduler_phase1 = lr_scheduler.CosineAnnealingLR(optimizer_phase1, T_max=EPOCHS_PHASE1, eta_min=LR_PHASE1/100)
    
    print(f"Optimizer for Phase 1 will train {sum(p.numel() for p in params_to_train_phase1_list)} parameters.")

    pollen_model, history_phase1 = train_model(
        pollen_model, dataloaders, dataset_sizes, criterion, optimizer_phase1, scheduler_phase1,
        num_epochs=EPOCHS_PHASE1, device=device, checkpoint_name=CHECKPOINT_PHASE1
    )
    plot_training_history(history_phase1, "Phase 1 Head Training", output_dir=output_dir) 
    print("--- Finished Training Phase 1 ---")

    # PHASE 2: Fine-tune the entire model
    print(f"\n--- Starting Training Phase 2: Full model fine-tuning ---")
    print(f"Epochs: {EPOCHS_PHASE2}, Backbone LR: {LR_BACKBONE_PHASE2}, Head LR: {LR_HEAD_PHASE2}")

    print(f"Loading best weights from Phase 1: {CHECKPOINT_PHASE1}")
    pollen_model.load_state_dict(torch.load(CHECKPOINT_PHASE1, map_location=device, weights_only=True))
    pollen_model.to(device) 

    for param in pollen_model.parameters():
        param.requires_grad = True

    head_params = []
    backbone_params = []
    for name, param in pollen_model.named_parameters():
        if 'head.fc' in name: 
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    if not head_params:
        print("Warning: 'head.fc' not found for differential LR. Applying backbone LR to all head params.")
        for name, param in pollen_model.named_parameters():
            if name.startswith('head'): 
                 head_params.append(param)
                 if param in backbone_params: 
                     backbone_params.remove(param)
    
    if not head_params: 
        print("ERROR: Could not separate head parameters for differential learning rates. Using single LR for all.")
        optimizer_phase2 = optim.AdamW(pollen_model.parameters(), lr=LR_BACKBONE_PHASE2, weight_decay=0.01)
    else:
        print(f"Phase 2: Training {sum(p.numel() for p in backbone_params)} backbone parameters with LR {LR_BACKBONE_PHASE2}")
        print(f"Phase 2: Training {sum(p.numel() for p in head_params)} head parameters with LR {LR_HEAD_PHASE2}")
        optimizer_phase2 = optim.AdamW([
            {'params': backbone_params, 'lr': LR_BACKBONE_PHASE2},
            {'params': head_params, 'lr': LR_HEAD_PHASE2}
        ], weight_decay=0.01)

    scheduler_phase2 = lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=EPOCHS_PHASE2, eta_min=LR_BACKBONE_PHASE2/10)

    pollen_model, history_phase2 = train_model(
        pollen_model, dataloaders, dataset_sizes, criterion, optimizer_phase2, scheduler_phase2,
        num_epochs=EPOCHS_PHASE2, device=device, checkpoint_name=CHECKPOINT_PHASE2
    )
    plot_training_history(history_phase2, "Phase 2 Full Fine-tuning", output_dir=output_dir) 
    print("--- Finished Training Phase 2 ---")

    # Final Evaluation on Test Set
    print("\n--- Starting Final Evaluation on Test Set ---")
    print(f"Loading best weights from Phase 2 for testing: {CHECKPOINT_PHASE2}")
    
    pollen_model.load_state_dict(torch.load(CHECKPOINT_PHASE2, map_location=device, weights_only=True))
    pollen_model.to(device)

    evaluate_model(pollen_model, dataloaders['test'], device, class_names, criterion, output_dir=output_dir) 

    print("\nFULL SCRIPT COMPLETE.")
