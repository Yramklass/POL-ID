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
import pandas as pd
from sklearn.metrics import classification_report

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
            # transforms.TrivialAugmentWide(), # Auto-augmentation- reduced accuracy
            transforms.RandomRotation(45), 
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        ]),
        # Keep validation and test transforms simple
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
    
    # Determine the classes to use from the 'train' directory
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
        
        # Filter all_class_names based on the requested list
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
        
    # Create the canonical (single source of truth) class_to_idx mapping 
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names_to_use)}
    num_classes = len(class_names_to_use)
    
    image_datasets = {}
    for phase in ['train', 'val', 'test']:
        phase_dir = Path(base_data_dir) / phase
        # Manually collect all image paths and their correct integer labels
        samples = []
        for class_name in class_names_to_use:
            class_dir = phase_dir / class_name
            if class_dir.is_dir():
                label = class_to_idx[class_name]
                for img_path in class_dir.glob('*'): # You might want to filter for .jpg, .png, etc.
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif']:
                        samples.append((str(img_path), label))

        # Create a dataset from the manually collected samples
        # We use a base DatasetFolder and provide our own loader and sample list
        # This ensures the labels are 100% consistent across all splits.
        dataset = datasets.DatasetFolder(
            root=str(phase_dir.parent), # Root is the base data dir
            loader=datasets.folder.default_loader, # Standard image loader
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

# ParallelFusionModel Definition 
class ParallelFusionModel(nn.Module):
    def __init__(self, num_classes, convnext_model_name='convnext_small', swin_model_name='swin_small_patch4_window7_224', pretrained=True):
        super().__init__()
        self.convnext_model_name = convnext_model_name
        self.swin_model_name = swin_model_name

        # Load ConvNext backbone
        # num_classes=0 ensures the classifier head is removed, returning features
        self.convnext_model = timm.create_model(convnext_model_name, pretrained=pretrained, num_classes=0)
        convnext_feature_dim = self.convnext_model.num_features

        # Load Swin Transformer backbone
        self.swin_model = timm.create_model(swin_model_name, pretrained=pretrained, num_classes=0)
        swin_feature_dim = self.swin_model.num_features

        # Fusion layer and classifier
        self.fusion_dim = convnext_feature_dim + swin_feature_dim
        self.fusion_classifier = nn.Linear(self.fusion_dim, num_classes)
        
        combined_features_dim = self.fusion_dim
        self.fusion_classifier = nn.Sequential(
            nn.Linear(combined_features_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5), 
            nn.Linear(1024, num_classes)
            )

        print(f"ParallelFusionModel initialized:")
        print(f"  ConvNext ({convnext_model_name}) output features: {convnext_feature_dim}")
        print(f"  Swin Transformer ({swin_model_name}) output features: {swin_feature_dim}")
        print(f"  Combined fusion features: {self.fusion_dim}")
        print(f"  Classifier for {num_classes} classes.")
        
    def forward(self, x, return_embeddings=False):
        """
        Modified forward pass.
        - If return_embeddings is False (default), it returns only the final classification logits.
        - If return_embeddings is True, it returns both the logits and the fused feature vector.
        """
        features_convnext = self.convnext_model(x)
        features_swin = self.swin_model(x)
        
        # Fused Feature Embedding
        fused_features = torch.cat((features_convnext, features_swin), dim=1)
        
        # Final classification
        output = self.fusion_classifier(fused_features)

        if return_embeddings:
            return output, fused_features
        else:
            return output

def create_parallel_fusion_model(num_classes, convnext_model_name='convnext_tiny', swin_model_name='swin_tiny_patch4_window7_224', pretrained=True):
    """
    Creates a Parallel Fusion model using ConvNext and Swin Transformer.
    """
    print(f"Creating Parallel Fusion model with ConvNext: {convnext_model_name}, Swin: {swin_model_name} for {num_classes} classes.")
    model = ParallelFusionModel(num_classes, convnext_model_name, swin_model_name, pretrained)
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
            if optimizer.param_groups:
                for i, param_group in enumerate(optimizer.param_groups):
                    pg_lr = param_group['lr']
                    pg_name = param_group.get('name', f'Group {i+1}') 
                    print(f'{phase.capitalize()} Current LR for {pg_name}: {pg_lr:.7f}')


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
    CLASS_LIST_FILE =  None

    script_location_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_location_dir, "training_outputs_parallel_fusion") 
    os.makedirs(output_dir, exist_ok=True)
    print(f"Checkpoints and plots will be saved in: {output_dir}")

    # Parallel Fusion Model Configuration
    CONVNEXT_MODEL_NAME = 'convnext_small'
    SWIN_MODEL_NAME = 'swin_small_patch4_window7_224' 

    BATCH_SIZE = 16
    NUM_WORKERS = 1 
    print(f"Using NUM_WORKERS = {NUM_WORKERS} for DataLoaders.")

    # Phase 1: Train only the fusion head
    LR_PHASE1_FUSION_HEAD = 1e-3
    EPOCHS_PHASE1 = 25 
    CHECKPOINT_PHASE1 = os.path.join(output_dir, 'pollen_parallel_fusion_phase1_head_best.pth')

    # Phase 2: Fine-tune the entire model (backbones + head)
    LR_PHASE2_CONVNEXT = 1e-4
    LR_PHASE2_SWIN = 1e-4
    LR_PHASE2_FUSION_HEAD = 5e-4
    EPOCHS_PHASE2 = 50 
    CHECKPOINT_PHASE2 = os.path.join(output_dir, 'pollen_parallel_fusion_phase2_full_best.pth')

    NUM_CLASSES_TO_USE = None # or None to use all
    # Load Data
    try:
    
        dataloaders, dataset_sizes, class_names, num_classes = load_data(
            base_data_dir, 
            batch_size=BATCH_SIZE, 
            img_size=IMG_SIZE, 
            num_workers=NUM_WORKERS,
            num_classes_to_use=NUM_CLASSES_TO_USE,
            class_list_file=CLASS_LIST_FILE 
        )
        print(f"Successfully loaded and filtered data for {num_classes} classes.")
    except Exception as e:
        print(f"Error during data loading: {e}")
        exit()


    # Create Parallel Fusion Model
    pollen_model = create_parallel_fusion_model(
        num_classes=num_classes,
        convnext_model_name=CONVNEXT_MODEL_NAME,
        swin_model_name=SWIN_MODEL_NAME,
        pretrained=True
    )

    # Setup Device & CUDA Diagnostics
    print(f"\n--- Device Setup & CUDA Diagnostics ---")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device index: {torch.cuda.current_device()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i} name: {torch.cuda.get_device_name(i)}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is NOT available. PyTorch will use CPU.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pollen_model.to(device)
    print(f"Parallel Fusion Model loaded on device: {device}")

    # Unweighted loss function
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Calculate class weights for weighted loss function 
    print("\nCalculating class weights to handle imbalance...")
    class_counts = np.bincount(dataloaders['train'].dataset.targets)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * num_classes # Normalize
    class_weights = class_weights.to(device)

    print(f"Class weights calculated and moved to {device}.")


    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Phase 1: Train the fusion classifier head 
    print(f"\n--- Starting Training Phase 1: Fine-tuning fusion classifier head ---")
    print(f"Epochs: {EPOCHS_PHASE1}, Fusion Head LR: {LR_PHASE1_FUSION_HEAD}")

    # Freeze backbones
    for param in pollen_model.convnext_model.parameters():
        param.requires_grad = False
    for param in pollen_model.swin_model.parameters():
        param.requires_grad = False
    # Unfreeze only the fusion classifier
    for param in pollen_model.fusion_classifier.parameters():
        param.requires_grad = True

    params_to_train_phase1_list = list(pollen_model.fusion_classifier.parameters())

    if not params_to_train_phase1_list:
        print("ERROR: No parameters selected for training in Phase 1. Check fusion_classifier parameters.")
        exit()

    optimizer_phase1 = optim.AdamW(
        [{'params': params_to_train_phase1_list, 'name': 'FusionHead'}], 
        lr=LR_PHASE1_FUSION_HEAD,
        weight_decay=0.01
    )
    scheduler_phase1 = lr_scheduler.CosineAnnealingLR(optimizer_phase1, T_max=EPOCHS_PHASE1, eta_min=LR_PHASE1_FUSION_HEAD/100)

    print(f"Optimizer for Phase 1 will train {sum(p.numel() for p in params_to_train_phase1_list)} parameters in the fusion head.")

    pollen_model, history_phase1 = train_model(
        pollen_model, dataloaders, dataset_sizes, criterion, optimizer_phase1, scheduler_phase1,
        num_epochs=EPOCHS_PHASE1, device=device, checkpoint_name=CHECKPOINT_PHASE1
    )
    plot_training_history(history_phase1, "Phase 1 Fusion Head Training", output_dir=output_dir)
    print("--- Finished Training Phase 1 ---")

    # Phase 2: Fine-tune the entire model 
    print(f"\n--- Starting Training Phase 2: Full parallel fusion model fine-tuning ---")
    print(f"Epochs: {EPOCHS_PHASE2}, ConvNext LR: {LR_PHASE2_CONVNEXT}, Swin LR: {LR_PHASE2_SWIN}, Fusion Head LR: {LR_PHASE2_FUSION_HEAD}")

    print(f"Loading best weights from Phase 1: {CHECKPOINT_PHASE1}")
    # Ensure to use map_location in case the model was saved on GPU and loading on CPU or vice-versa
    pollen_model.load_state_dict(torch.load(CHECKPOINT_PHASE1, map_location=device, weights_only=True))
    pollen_model.to(device)

    # Unfreeze all parts for fine-tuning
    for param in pollen_model.convnext_model.parameters():
        param.requires_grad = True
    for param in pollen_model.swin_model.parameters():
        param.requires_grad = True
    for param in pollen_model.fusion_classifier.parameters():
        param.requires_grad = True

    # Parameters for differential learning rates
    convnext_params = list(pollen_model.convnext_model.parameters())
    swin_params = list(pollen_model.swin_model.parameters())
    fusion_head_params = list(pollen_model.fusion_classifier.parameters())

    optimizer_phase2 = optim.AdamW([
        {'params': convnext_params, 'lr': LR_PHASE2_CONVNEXT, 'name': 'ConvNext_Backbone'},
        {'params': swin_params, 'lr': LR_PHASE2_SWIN, 'name': 'Swin_Backbone'},
        {'params': fusion_head_params, 'lr': LR_PHASE2_FUSION_HEAD, 'name': 'Fusion_Head'}
    ], weight_decay=0.01)

    # Use the smallest backbone LR for eta_min reference in scheduler
    min_backbone_lr_phase2 = min(LR_PHASE2_CONVNEXT, LR_PHASE2_SWIN)
    scheduler_phase2 = lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=EPOCHS_PHASE2, eta_min=min_backbone_lr_phase2/10)

    print(f"Phase 2: Training {sum(p.numel() for p in convnext_params)} ConvNext backbone parameters.")
    print(f"Phase 2: Training {sum(p.numel() for p in swin_params)} Swin backbone parameters.")
    print(f"Phase 2: Training {sum(p.numel() for p in fusion_head_params)} fusion head parameters.")

    pollen_model, history_phase2 = train_model(
        pollen_model, dataloaders, dataset_sizes, criterion, optimizer_phase2, scheduler_phase2,
        num_epochs=EPOCHS_PHASE2, device=device, checkpoint_name=CHECKPOINT_PHASE2
    )
    plot_training_history(history_phase2, "Phase 2 Full Fusion Fine-tuning", output_dir=output_dir)
    print("--- Finished Training Phase 2 ---")
    
    # Embed class names before saving final model
    pollen_model.class_names = class_names

    # Save full model with metadata
    torch.save(pollen_model, os.path.join(output_dir, 'pollen_parallel_fusion_final_full.pth'))


    # Final Evaluation on Test Set
    print("\n--- Starting Final Evaluation on Test Set ---")
    print(f"Loading best weights from Phase 2 for testing: {CHECKPOINT_PHASE2}")

    
    pollen_model.load_state_dict(torch.load(CHECKPOINT_PHASE2, map_location=device, weights_only=True))
    pollen_model.to(device)

    evaluate_model(pollen_model, dataloaders['test'], device, class_names, criterion, output_dir=output_dir)


    print("\nFULL PARALLEL FUSION SCRIPT COMPLETE.")
  