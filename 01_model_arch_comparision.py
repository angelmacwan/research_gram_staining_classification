import os
import shutil
import random
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score
import numpy as np
import pandas as pd
from torch.amp import autocast, GradScaler
import gc
import logging
from datetime import datetime

# Configure logging
log_filename = f"01_model_arch_comparision_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)


# Set random seeds for reproducibility
SEED = 12
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------- CONFIG --------------------
# DATA SET PATHS
train_data_dir = "./TRAIN"
test_data_dir = "./TEST"

batch_size = 45
epochs = 25
num_workers = 4  # Parallel data loading (adjust based on CPU cores)
use_amp = True  # Automatic Mixed Precision for faster training
samples_per_class = 9999 # Set to a number to limit samples per class, or None to use all samples

# Learning rate scheduler settings
use_scheduler = True  # Enable/disable learning rate scheduler
scheduler_type = "cosine"  # Options: "cosine", "step", "plateau"
initial_lr = 1e-4  # Initial learning rate
min_lr = 1e-6  # Minimum learning rate for cosine scheduler
step_size = 3  # For step scheduler: decay LR every N epochs
gamma = 0.1  # For step scheduler: multiply LR by this factor

# Early stopping settings
use_early_stopping = True  # Enable/disable early stopping
early_stop_patience = 3  # Number of epochs to wait for improvement before stopping
early_stop_min_delta = 0.001  # Minimum change in F1 score to qualify as improvement



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


# Device selection: CUDA GPU or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Check if multiple GPUs are available
    num_gpus = torch.cuda.device_count()
    use_multi_gpu = num_gpus > 1
    logger.info(f"Found {num_gpus} GPU(s)")
else:
    device = torch.device("cpu")
    use_multi_gpu = False
    use_amp = False  # AMP only works on CUDA
logger.info(f"Using device: {device}")
logger.info(f"Multi-GPU training: {use_multi_gpu}")
logger.info(f"Mixed Precision: {use_amp}")

model_names = [
    "tiny_vit_5m_224",
    "volo_d1_224",
    "convformer_s36",
    "convnextv2_tiny",
    "tiny_vit_11m_224",
    "convit_base",
    "inception_next_small",
    "visformer_small",
    "convmixer_768_32",
    "crossvit_small_240",
    "visformer_tiny",
    "vgg19",
    "efficientformerv2_s0",
    "beitv2_base_patch16_224",
    "efficientvit_b0",
    "deit3_base_patch16_224",
    "densenet121",
    "inception_resnet_v2",
    "deit_tiny_patch16_224",
    "resnext50_32x4d",
    "crossvit_9_240",
    "tinynet_a",
    "fastvit_s12",
    "efficientnet_b0",
]

# -------------------------------------------------

# Transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load datasets
train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)

# Limit samples per class if specified
if samples_per_class is not None:
    from torch.utils.data import Subset
    
    # Group indices by class
    class_indices = {}
    for idx, (_, label) in enumerate(train_dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Select up to n samples per class
    selected_indices = []
    for label, indices in class_indices.items():
        # If n is bigger than total samples, use all samples
        n_samples = min(samples_per_class, len(indices))
        selected_indices.extend(random.sample(indices, n_samples))
    
    # Create subset
    train_dataset = Subset(train_dataset, selected_indices)
    logger.info(f"Limited training set to {samples_per_class} samples per class (or all available if fewer)")

# Optimized DataLoaders with parallel workers and pinned memory
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True if device.type == 'cuda' else False,
    persistent_workers=True if num_workers > 0 else False
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True if device.type == 'cuda' else False,
    persistent_workers=True if num_workers > 0 else False
)

logger.info(f"Train set size: {len(train_dataset)}")
logger.info(f"Test set size: {len(test_dataset)}")
# Access classes from the underlying dataset if it's a Subset
if hasattr(train_dataset, 'dataset'):
    logger.info(f"Number of classes: {len(train_dataset.dataset.classes)}")
    logger.info(f"Classes: {train_dataset.dataset.classes}")
else:
    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    logger.info(f"Classes: {train_dataset.classes}")

model_stats = []

# Training + evaluation loop
def train_and_eval(model_name):
    # Load model
    # Get number of classes from the underlying dataset if it's a Subset
    num_classes = len(train_dataset.dataset.classes) if hasattr(train_dataset, 'dataset') else len(train_dataset.classes)
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model.to(device)
    
    # Enable multi-GPU training if available
    if use_multi_gpu:
        logger.info(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Compile model for faster execution (PyTorch 2.0+)
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile()")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")

    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # Learning rate scheduler
    scheduler = None
    if use_scheduler:
        if scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
            logger.info(f"Using CosineAnnealingLR scheduler (min_lr={min_lr})")
        elif scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            logger.info(f"Using StepLR scheduler (step_size={step_size}, gamma={gamma})")
        elif scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=2, verbose=True)
            logger.info("Using ReduceLROnPlateau scheduler")
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None

    # Track best metrics
    best_f1 = 0.0
    best_accuracy = 0.0
    best_precision = 0.0
    best_epoch = 0
    
    # Early stopping variables
    epochs_without_improvement = 0
    early_stopped = False

    # Train on entire training set
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # Mixed precision training
            if use_amp:
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        avg_loss = running_loss/len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, LR: {current_lr:.6f}")
        
        # Evaluate on test set after each epoch
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                if use_amp:
                    with autocast(device_type='cuda'):
                        outputs = model(images)
                else:
                    outputs = model(images)
                    
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Calculate metrics on test set
        test_acc = accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred, average="macro")
        test_precision = precision_score(y_true, y_pred, average="macro")
        
        logger.debug(f"Test Metrics - Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, F1: {test_f1:.4f}")
        
        # Track best F1 score
        if test_f1 > best_f1 + early_stop_min_delta:
            best_f1 = test_f1
            best_accuracy = test_acc
            best_precision = test_precision
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            logger.warning(f"*** New best F1 score: {best_f1:.4f} at epoch {best_epoch} ***")
        else:
            epochs_without_improvement += 1
            if use_early_stopping:
                logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        # Check early stopping condition
        if use_early_stopping and epochs_without_improvement >= early_stop_patience:
            logger.warning(f"\n*** Early stopping triggered after {epoch + 1} epochs (no improvement for {early_stop_patience} epochs) ***")
            early_stopped = True
            break
        
        # Step the scheduler
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()

    # Save the best metrics
    if early_stopped:
        logger.warning(f"Training stopped early. Best Results - Epoch: {best_epoch}, Accuracy: {best_accuracy:.4f}, Precision: {best_precision:.4f}, F1: {best_f1:.4f}\n")
    else:
        logger.info(f"Training completed all epochs. Best Results - Epoch: {best_epoch}, Accuracy: {best_accuracy:.4f}, Precision: {best_precision:.4f}, F1: {best_f1:.4f}\n")
    
    model_stats.append(
        {"model_name": model_name, "best_epoch": best_epoch, "accuracy": best_accuracy, "precision": best_precision, "f1": best_f1, "early_stopped": early_stopped}
    )
    
    # Clean up memory
    del model, optimizer, scaler

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()


# Run for all models
for i in range(len(model_names)):
    logger.info(f"Training model {i+1} of {len(model_names)}")
    logger.info(model_names[i])
    train_and_eval(model_names[i])


# Save to CSV
df = pd.DataFrame(model_stats)
df.to_csv("model_comparision_results.csv", index=False)

logger.info("TRAINING COMPLETED")

logger.info(f"Final Results:\n{df}")