import os
import gc
import time
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime

# Configure logging
log_filename = f"04_final_model_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

SEED = 16
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.set_float32_matmul_precision('high')

# Prefer Apple Silicon MPS (Metal) if available, then CUDA, else CPU
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
    device = torch.device("mps")
elif torch.cuda.is_available(): 
    device = torch.device("cuda")
else: 
    device = torch.device("cpu")

logger.info(f"Using device: {device}")


# -------------------- CONFIG --------------------
model_name = 'volo_d5_512'

# DATA SET PATHS
train_data_dir = "./TRAIN"
test_data_dir = "./TEST"

epochs = 99
drop_rate = 0.3

# Learning rate scheduler settings
use_scheduler = True
scheduler_type = "cosine"  # Options: "cosine", "step", "plateau"
initial_lr = 1e-4
min_lr = 1e-6
step_size = 3
gamma = 0.1

# Early stopping settings
use_early_stopping = True
early_stop_patience = 5
early_stop_min_delta = 0.001

# Mixed precision training
use_mixed_precision = True


# Enable multi-GPU training if available
use_multi_gpu = torch.cuda.device_count() > 1 if device.type == 'cuda' else False
logger.info(f"Multi-GPU training: {use_multi_gpu}")



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


def calculate_comprehensive_metrics(y_true, y_pred, class_names):
    """
    Calculate comprehensive evaluation metrics including accuracy, F1, precision, recall,
    sensitivity (per class), and specificity (per class).
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    
    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None)
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    
    # Confusion matrix for sensitivity and specificity calculation
    conf_matrix = confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)
    
    sensitivity = np.zeros(num_classes)  # Same as recall
    specificity = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = conf_matrix[i, i]  # True Positives
        fn = np.sum(conf_matrix[i, :]) - tp  # False Negatives
        fp = np.sum(conf_matrix[:, i]) - tp  # False Positives
        tn = np.sum(conf_matrix) - tp - fn - fp  # True Negatives
        
        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Specificity = TN / (TN + FP)
        specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Create comprehensive metrics dictionary
    metrics = {
        'overall_metrics': {
            'accuracy': float(accuracy),
            'f1_weighted': float(f1_weighted),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'precision_macro': float(precision_macro),
            'recall_weighted': float(recall_weighted),
            'recall_macro': float(recall_macro)
        },
        'per_class_metrics': {}
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        metrics['per_class_metrics'][class_name] = {
            'f1_score': float(f1_per_class[i]),
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'sensitivity': float(sensitivity[i]),
            'specificity': float(specificity[i])
        }
    
    # Add confusion matrix
    metrics['confusion_matrix'] = conf_matrix.tolist()
    metrics['class_names'] = class_names
    
    return metrics


def find_optimal_batch_size(model, input_shape, device, start_batch=1, safety_factor=0.8):
    model.to(device)
    model.train()
    
    # Clear cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Create dummy input with batch_size=1
    dummy_input = torch.randn(start_batch, *input_shape).to(device)
    
    # Forward + backward pass to measure memory
    try:
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            
        outputs = model(dummy_input)
        loss = outputs.sum()
        loss.backward()
        
        if device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            # Estimate memory per image (linear scaling assumption)
            memory_per_image = memory_used / start_batch
            available_memory = total_memory * safety_factor
            optimal_batch = int(available_memory / memory_per_image)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"VRAM Analysis:")
            logger.info(f"  Total VRAM: {total_memory:.2f} GB")
            logger.info(f"  Used for batch={start_batch}: {memory_used:.2f} GB")
            logger.info(f"  Memory per image: {memory_per_image:.3f} GB")
            logger.info(f"  Available (with {safety_factor:.0%} safety): {available_memory:.2f} GB")
            logger.info(f"  Calculated optimal batch size: {optimal_batch}")
            logger.info(f"{'='*60}\n")
    except RuntimeError as e:
        logger.info(f"Error during memory test: {e}")
        optimal_batch = 1
        
    finally:
        # Cleanup
        del dummy_input, outputs, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    return max(1, optimal_batch)




# Get model-specific configuration and optimal batch size
m = timm.create_model(model_name, pretrained=True, num_classes=4, drop_rate=drop_rate)
model_info = m.default_cfg

input_shape = model_info['input_size']
transform_mean = model_info['mean']
transform_std = model_info['std']
batch_size = find_optimal_batch_size(m, input_shape, device, start_batch=1, safety_factor=0.88)

logger.info(f"USING MODEL ARCHITECTURE {model_info['architecture']}")
logger.info(f"INPUT SHAPE = {input_shape}")
logger.info(f"       MEAN = {transform_mean}")
logger.info(f"        STD = {transform_std}")
logger.info(f" BATCH SIZE = {batch_size}")

del m

# Create robust model-specific transforms
train_transform = transforms.Compose([
    transforms.Resize(input_shape[1:]),
    # Geometric augmentations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),  # Microscopy images can be rotated in any direction
    transforms.RandomRotation(degrees=180),  # Full rotation for microscopy
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.1, 0.1),  # Small translations
        scale=(0.9, 1.1),      # Scale variations
        shear=5                # Small shear
    ),
    # Color/intensity augmentations (important for staining variations)
    transforms.ColorJitter(
        brightness=0.2,        # Brightness variations in staining
        contrast=0.2,          # Contrast variations
        saturation=0.2,        # Saturation changes in staining
        hue=0.05              # Small hue variations
    ),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    ], p=0.2),  # Occasional blur to simulate focus variations
    # Convert to tensor and normalize
    transforms.ToTensor(),
    # Random noise to improve robustness
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01 if torch.rand(1) < 0.1 else x),
    transforms.Normalize(mean=transform_mean, std=transform_std),
    # Random erasing to improve generalization
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
])

test_transform = transforms.Compose([
    transforms.Resize(input_shape[1:]),
    transforms.ToTensor(),
    transforms.Normalize(mean=transform_mean, std=transform_std),
])

# Load datasets with model-specific transforms
train_dataset = datasets.ImageFolder(root=train_data_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=test_transform)

logger.info("TRAIN SET CLASS MAPPING")
logger.info(train_dataset.class_to_idx)
logger.info("TEST SET CLASS MAPPING")
logger.info(test_dataset.class_to_idx)

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4,
    pin_memory=True if device.type == 'cuda' else False,
    persistent_workers=True
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=4,
    pin_memory=True if device.type == 'cuda' else False,
    persistent_workers=True
)

# Load model
model = timm.create_model(
    model_name, pretrained=True, num_classes=len(train_dataset.classes), drop_rate=drop_rate
)

model.to(device)

# Enable multi-GPU training if available
if use_multi_gpu:
    model = nn.DataParallel(model)

# Compile model for faster execution (PyTorch 2.0+)
if hasattr(torch, 'compile') and device.type == 'cuda':
    model = torch.compile(model)
    logger.info("Model compiled with torch.compile()")

criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=initial_lr)

# Learning rate scheduler
scheduler = None
if use_scheduler:
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=2, verbose=True)

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and device.type == 'cuda' else None

best_acc = 0.0
best_f1 = 0.0
best_epoch = 0
epochs_without_improvement = 0
early_stopped = False

# Train
training_start_time = time.time()
for epoch in range(epochs):
    epoch_start_time = time.time()
    logger.info(f'Epoch {epoch+1}/{epochs}')
    
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        if use_mixed_precision and scaler is not None:
            with torch.amp.autocast('cuda'):
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
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total
    avg_loss = running_loss/len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            if use_mixed_precision and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    val_acc = correct / total
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    val_loss_avg = val_loss/len(test_loader)
    
    epoch_time = time.time() - epoch_start_time
    logger.info(f'Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss_avg:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s')

    # Check for improvement and save best model
    if val_f1 > best_f1 + early_stop_min_delta:
        best_acc = val_acc
        best_f1 = val_f1
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        
        # Save the best model
        if use_multi_gpu:
            torch.save(model.module.state_dict(), f'{model_name}_best.pth')
        else:
            torch.save(model.state_dict(), f'{model_name}_best.pth')
        logger.info(f"*** New best model saved! F1: {best_f1:.4f} ***")
    else:
        epochs_without_improvement += 1

    # Check early stopping condition
    if use_early_stopping and epochs_without_improvement >= early_stop_patience:
        logger.info(f"\n*** Early stopping triggered after {epoch + 1} epochs (no improvement for {early_stop_patience} epochs) ***")
        early_stopped = True
        break

    # Step the scheduler
    if scheduler is not None:
        if scheduler_type == "plateau":
            scheduler.step(val_loss_avg)
        else:
            scheduler.step()

total_training_time = time.time() - training_start_time

logger.info(f'\nTraining completed!')
logger.info(f'Best Validation Accuracy: {best_acc:.4f}')
logger.info(f'Best Validation F1: {best_f1:.4f}')
logger.info(f'Best epoch: {best_epoch}')
logger.info(f'Early stopped: {early_stopped}')
logger.info(f'Total training time: {total_training_time:.2f}s')

# Load best model for final evaluation
if use_multi_gpu:
    model.module.load_state_dict(torch.load(f'{model_name}_best.pth'))
else:
    model.load_state_dict(torch.load(f'{model_name}_best.pth'))

logger.info(f"\nLoaded best model for final evaluation...")

# Final evaluation on test set
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        if use_mixed_precision and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(images)
        else:
            outputs = model(images)
            
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

logger.info("\nFinal Test Results:")
logger.info(classification_report(y_true, y_pred, target_names=test_dataset.classes))

# Calculate comprehensive metrics
comprehensive_metrics = calculate_comprehensive_metrics(y_true, y_pred, test_dataset.classes)

# Display comprehensive metrics
logger.info("\n" + "="*80)
logger.info("COMPREHENSIVE EVALUATION METRICS")
logger.info("="*80)

logger.info(f"\nOVERALL METRICS:")
logger.info(f"  Accuracy: {comprehensive_metrics['overall_metrics']['accuracy']:.4f}")
logger.info(f"  F1 Score (Weighted): {comprehensive_metrics['overall_metrics']['f1_weighted']:.4f}")
logger.info(f"  F1 Score (Macro): {comprehensive_metrics['overall_metrics']['f1_macro']:.4f}")
logger.info(f"  Precision (Weighted): {comprehensive_metrics['overall_metrics']['precision_weighted']:.4f}")
logger.info(f"  Precision (Macro): {comprehensive_metrics['overall_metrics']['precision_macro']:.4f}")
logger.info(f"  Recall (Weighted): {comprehensive_metrics['overall_metrics']['recall_weighted']:.4f}")
logger.info(f"  Recall (Macro): {comprehensive_metrics['overall_metrics']['recall_macro']:.4f}")

logger.info(f"\nPER-CLASS METRICS:")
for class_name in test_dataset.classes:
    metrics = comprehensive_metrics['per_class_metrics'][class_name]
    logger.info(f"  {class_name}:")
    logger.info(f"    F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"    Precision: {metrics['precision']:.4f}")
    logger.info(f"    Recall: {metrics['recall']:.4f}")
    logger.info(f"    Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"    Specificity: {metrics['specificity']:.4f}")

logger.info(f"\nCONFUSION MATRIX:")
conf_matrix = np.array(comprehensive_metrics['confusion_matrix'])
logger.info("Classes:", test_dataset.classes)
logger.info(conf_matrix)

# Save metrics to files
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_results_dir = f"{model_name}_results_{timestamp}"
os.makedirs(model_results_dir, exist_ok=True)

# Save comprehensive metrics as JSON
metrics_file = os.path.join(model_results_dir, f"{model_name}_comprehensive_metrics.json")
with open(metrics_file, 'w') as f:
    json.dump(comprehensive_metrics, f, indent=2)

# Save metrics as CSV for easier analysis
# Overall metrics CSV
overall_df = pd.DataFrame([comprehensive_metrics['overall_metrics']])
overall_csv = os.path.join(model_results_dir, f"{model_name}_overall_metrics.csv")
overall_df.to_csv(overall_csv, index=False)

# Per-class metrics CSV
per_class_data = []
for class_name, metrics in comprehensive_metrics['per_class_metrics'].items():
    row = {'class_name': class_name}
    row.update(metrics)
    per_class_data.append(row)

per_class_df = pd.DataFrame(per_class_data)
per_class_csv = os.path.join(model_results_dir, f"{model_name}_per_class_metrics.csv")
per_class_df.to_csv(per_class_csv, index=False)

# Save confusion matrix as CSV
conf_matrix_df = pd.DataFrame(conf_matrix, 
                             index=[f"True_{cls}" for cls in test_dataset.classes],
                             columns=[f"Pred_{cls}" for cls in test_dataset.classes])
conf_matrix_csv = os.path.join(model_results_dir, f"{model_name}_confusion_matrix.csv")
conf_matrix_df.to_csv(conf_matrix_csv)

# Save training summary
training_summary = {
    'model_name': model_name,
    'total_training_time': total_training_time,
    'best_epoch': best_epoch,
    'best_validation_accuracy': best_acc,
    'best_validation_f1': best_f1,
    'early_stopped': early_stopped,
    'final_test_accuracy': comprehensive_metrics['overall_metrics']['accuracy'],
    'final_test_f1_weighted': comprehensive_metrics['overall_metrics']['f1_weighted'],
    'timestamp': timestamp,
    'device_used': str(device),
    'batch_size': batch_size,
    'epochs_trained': best_epoch if early_stopped else epochs,
    'train_dataset_path': train_data_dir,
    'test_dataset_path': test_data_dir
}

training_summary_file = os.path.join(model_results_dir, f"{model_name}_training_summary.json")
with open(training_summary_file, 'w') as f:
    json.dump(training_summary, f, indent=2)

logger.info(f"\n" + "="*80)
logger.info("RESULTS SAVED")
logger.info("="*80)
logger.info(f"Results directory: {model_results_dir}")
logger.info(f"1. Comprehensive metrics (JSON): {metrics_file}")
logger.info(f"2. Overall metrics (CSV): {overall_csv}")
logger.info(f"3. Per-class metrics (CSV): {per_class_csv}")
logger.info(f"4. Confusion matrix (CSV): {conf_matrix_csv}")
logger.info(f"5. Training summary (JSON): {training_summary_file}")
logger.info(f"6. Best model: {model_name}_best.pth")

# Clean up memory
if device.type == 'cuda':
    torch.cuda.empty_cache()
gc.collect()

logger.info(f"\nTraining completed successfully!")
logger.info(f"Best model saved as: {model_name}_best.pth")
logger.info(f"All evaluation metrics saved in: {model_results_dir}")
logger.info("="*80)