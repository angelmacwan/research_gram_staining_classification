'''
After selecting a model architecture in step 01, we run variants of that model.
These models are evaluated on the combined dataset.

Input : Combined dataset (SET 01 + SET 02) images
Output : comparison of model variants
'''

import os
import sys
import pandas as pd
import time
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score
from datetime import datetime
import logging

# Configure logging
log_filename = f"02_model_selection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

# -------------------- CONFIG --------------------
model_list = [
    "volo_d5_512",
    "volo_d1_384",
    "volo_d2_384",
    "volo_d3_448",
    "volo_d4_448",
]

# DATA SET PATHS
train_data_dir = "./TRAIN"
test_data_dir = "./TEST"

epochs = 25

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


# Device selection: CUDA GPU or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    use_multi_gpu = num_gpus > 1
    logger.info(f"Found {num_gpus} GPU(s)")
else:
    device = torch.device("cpu")
    use_multi_gpu = False
logger.info(f"Using device: {device}")
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

def find_optimal_batch_size(model, input_shape, device, start_batch=1, safety_factor=0.8):
    import gc
    
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


# -------------------------------------------------

model_stats = []

# Training + evaluation loop
def train_and_eval(model_name):
    logger.info(f"\n{'#'*60}")
    logger.info(f"Training {model_name}...")
    logger.info(f"{'#'*60}")

    # Get model-specific configuration
    m = timm.create_model(model_name, pretrained=True, num_classes=4)
    model_info = m.default_cfg
    
    input_shape = model_info['input_size']
    transform_mean = model_info['mean']
    transform_std = model_info['std']

    logger.info(f"Model input shape: {input_shape}")
    logger.info(f"Model mean: {transform_mean}")
    logger.info(f"Model std: {transform_std}")

    # Calculate optimal batch size
    batch_size = find_optimal_batch_size(m, input_shape, device, start_batch=1, safety_factor=0.88)
     
    del m  # Clean up the temporary model

    # Create model-specific transforms
    train_transform = transforms.Compose([
        transforms.Resize(input_shape[1:]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std),
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

    # Optimized DataLoaders with parallel workers and pinned memory
    num_workers = 4  # Parallel data loading (adjust based on CPU cores)
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

    # Load model
    model = timm.create_model(
        model_name, pretrained=True, num_classes=len(train_dataset.classes)
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
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        elif scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=2, verbose=True)

    # Track best metrics
    best_f1 = 0.0
    best_accuracy = 0.0
    best_precision = 0.0
    best_epoch = 0
    
    # Early stopping variables
    epochs_without_improvement = 0
    early_stopped = False
    
    # Track epoch timings
    epoch_times = []

    # Train
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
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
        y_true_epoch, y_pred_epoch = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                y_true_epoch.extend(labels.cpu().numpy())
                y_pred_epoch.extend(preds.cpu().numpy())

        # Calculate metrics on test set
        test_acc = accuracy_score(y_true_epoch, y_pred_epoch)
        test_f1 = f1_score(y_true_epoch, y_pred_epoch, average="macro")
        test_precision = precision_score(y_true_epoch, y_pred_epoch, average="macro")
        
        logger.info(f"Test Metrics - Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, F1: {test_f1:.4f}")
        
        # Track best F1 score
        if test_f1 > best_f1 + early_stop_min_delta:
            best_f1 = test_f1
            best_accuracy = test_acc
            best_precision = test_precision
            best_epoch = epoch + 1
            epochs_without_improvement = 0
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
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        
        # Track epoch timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

    end_time = time.time()
    training_time = end_time - start_time
    avg_time_per_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0

    # Save the best metrics
    if early_stopped:
        logger.info(f"Training stopped early. Best Results - Epoch: {best_epoch}, Accuracy: {best_accuracy:.4f}, Precision: {best_precision:.4f}, F1: {best_f1:.4f}\n")
    else:
        logger.info(f"Training completed all epochs. Best Results - Epoch: {best_epoch}, Accuracy: {best_accuracy:.4f}, Precision: {best_precision:.4f}, F1: {best_f1:.4f}\n")

    model_stats.append(
        {"model_name": model_name, "best_epoch": best_epoch, "accuracy": best_accuracy, "precision": best_precision, "f1": best_f1, "training_time": training_time, "avg_time_per_epoch": avg_time_per_epoch, "early_stopped": early_stopped}
    )

    # Clean up memory
    del model, train_dataset, test_dataset, train_loader, test_loader

    if device.type == 'cuda':
        torch.cuda.empty_cache()


# Run for all models
for m in model_list:
    train_and_eval(m)

# print output
logger.info("\n\n\nmodel_name, best_epoch, accuracy, precision, f1, training_time, avg_time_per_epoch, early_stopped")
for i in model_stats:
    logger.info(
        f"{i['model_name']}, {i['best_epoch']}, {i['accuracy']:.4f}, {i['precision']:.4f}, {i['f1']:.4f}, {i['training_time']:.2f}s, {i['avg_time_per_epoch']:.2f}s, {i['early_stopped']}"
    )


# save to CSV
df = pd.DataFrame(model_stats)
csv_file = f"02_model_selection.csv"
df.to_csv(csv_file, index=False)

