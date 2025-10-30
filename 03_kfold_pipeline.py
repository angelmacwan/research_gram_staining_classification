import os
from collections import defaultdict
import gc
import time
import logging
from datetime import datetime

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

import pandas as pd
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    classification_report,
    recall_score
)

from sklearn.model_selection import StratifiedKFold

# Configure logging
log_filename = f"03_kfold_pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

# Check for CUDA device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(DEVICE)

# Enable multi-GPU training if available
use_multi_gpu = torch.cuda.device_count() > 1 if DEVICE == 'cuda' else False
logger.info(f"Multi-GPU training: {use_multi_gpu}")

# DATA SET PATH
train_data_dir = "./TRAIN"


# PARAMS
model_name = 'volo_d5_512'
num_epochs = 99 ## rely on early stopping
num_classes = 4
k_folds = 5
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
early_stop_patience = 4
early_stop_min_delta = 0.001

# Mixed precision training
use_mixed_precision = True


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


# Download the model
m = timm.create_model(model_name, pretrained=True, num_classes=num_classes, drop_rate=drop_rate)
model_info = m.default_cfg

input_shape = model_info['input_size']
transform_mean = model_info['mean']
transform_std = model_info['std']
batch_size = find_optimal_batch_size(m, input_shape, torch.device(DEVICE), start_batch=1, safety_factor=0.88)

logger.info(f"USING MODEL ARCHITECTURE {model_info['architecture']} ")
logger.info(f"INPUT SHAPE = {input_shape}")
logger.info(f"       MEAN = {transform_mean}")
logger.info(f"        STD = {transform_std}")
logger.info(f" BATCH SIZE = {batch_size}")

del m


transform = transforms.Compose([
        transforms.Resize(input_shape[1:]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std)
    ])

dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)

logger.info("CLASS MAPPING TRAIN")
logger.info(dataset.class_to_idx)


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
    

kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)

# Store results for each fold
fold_results = []

# K-Fold Loop
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset, dataset.targets)):
    logger.info('\n--------------------------------')
    logger.info(f'FOLD {fold+1}/{k_folds}')
    logger.info('--------------------------------')

    # Create data loaders for this fold
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=train_subsampler, 
        num_workers=4,
        pin_memory=True if DEVICE == 'cuda' else False,
        persistent_workers=True
    )
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=val_subsampler, 
        num_workers=4,
        pin_memory=True if DEVICE == 'cuda' else False,
        persistent_workers=True
    )

    # Initialize the model for this fold
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, drop_rate=drop_rate)
    model = model.to(DEVICE)
    
    # Enable multi-GPU training if available
    if use_multi_gpu:
        model = nn.DataParallel(model)
    
    # Compile model for faster execution (PyTorch 2.0+)
    if hasattr(torch, 'compile') and DEVICE == 'cuda':
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile()")
    
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    
    # Learning rate scheduler
    scheduler = None
    if use_scheduler:
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=2, verbose=True)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and DEVICE == 'cuda' else None

    best_acc = 0.0
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False

    # Model training loop START
    fold_start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            if use_mixed_precision and scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
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

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                
                if use_mixed_precision and DEVICE == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_acc = correct / total
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_precision = precision_score(all_labels, all_preds, average='weighted')
        val_recall = recall_score(all_labels, all_preds, average='weighted')
        val_loss_avg = val_loss/len(val_loader)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f'Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss_avg:.4f}, Val Accuracy: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s')

        # Check for improvement
        if val_f1 > best_f1 + early_stop_min_delta:
            best_acc = val_acc
            best_f1 = val_f1
            best_precision = val_precision
            best_recall = val_recall
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
                scheduler.step(val_loss_avg)
            else:
                scheduler.step()
    
    fold_time = time.time() - fold_start_time
    
    logger.info(f'Best Val Accuracy for fold {fold+1}: {best_acc:.4f}')
    logger.info(f'Best Val Precision for fold {fold+1}: {best_precision:.4f}')
    logger.info(f'Best Val Recall for fold {fold+1}: {best_recall:.4f}')
    logger.info(f'Best Val F1 for fold {fold+1}: {best_f1:.4f}')
    logger.info(f'Best epoch: {best_epoch}')
    logger.info(f'Early stopped: {early_stopped}')
    logger.info(f'Fold training time: {fold_time:.2f}s')
    
    fold_results.append({
        'fold': fold+1,
        'best_epoch': best_epoch,
        'test_accuracy': best_acc,
        'test_precision': best_precision,
        'test_recall': best_recall,
        'test_f1': best_f1,
        'early_stopped': early_stopped,
        'training_time': fold_time
    })
    
    # Clean up memory
    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

# Save fold training results
fold_df = pd.DataFrame(fold_results)
fold_df.to_csv('LOGS/03_kfold_training_results.csv', index=False)
logger.info("\nFold Training Results:")
logger.info(fold_df)

# Calculate and print overall K-fold results
logger.info(f"\nK-Fold Cross-Validation Summary ({k_folds} folds):")
logger.info(f"Average Best Validation Accuracy: {fold_df['test_accuracy'].mean():.4f} ± {fold_df['test_accuracy'].std():.4f}")
logger.info(f"Average Best Validation Precision: {fold_df['test_precision'].mean():.4f} ± {fold_df['test_precision'].std():.4f}")
logger.info(f"Average Best Validation Recall: {fold_df['test_recall'].mean():.4f} ± {fold_df['test_recall'].std():.4f}")
logger.info(f"Average Best Validation F1: {fold_df['test_f1'].mean():.4f} ± {fold_df['test_f1'].std():.4f}")
logger.info(f"Average Best Epoch: {fold_df['best_epoch'].mean():.1f} ± {fold_df['best_epoch'].std():.1f}")
logger.info(f"Average Training Time: {fold_df['training_time'].mean():.2f}s ± {fold_df['training_time'].std():.2f}s")
logger.info(f"Early Stopping Rate: {fold_df['early_stopped'].sum()}/{k_folds} folds ({100*fold_df['early_stopped'].mean():.1f}%)")

# Create final summary
summary_stats = {
    'k_folds': k_folds,
    'model_name': model_name,
    'avg_test_accuracy': fold_df['test_accuracy'].mean(),
    'std_test_accuracy': fold_df['test_accuracy'].std(),
    'avg_test_precision': fold_df['test_precision'].mean(),
    'std_test_precision': fold_df['test_precision'].std(),
    'avg_test_recall': fold_df['test_recall'].mean(),
    'std_test_recall': fold_df['test_recall'].std(),
    'avg_test_f1': fold_df['test_f1'].mean(),
    'std_test_f1': fold_df['test_f1'].std(),
    'avg_best_epoch': fold_df['best_epoch'].mean(),
    'avg_training_time': fold_df['training_time'].mean(),
    'early_stopping_rate': fold_df['early_stopped'].mean(),
    'total_training_time': fold_df['training_time'].sum()
}

# Save final summary
summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv('LOGS/03_kfold_summary.csv', index=False)
logger.info("K-Fold validation completed successfully!")
logger.info("Results saved to:")
logger.info("  - LOGS/03_kfold_training_results.csv (detailed results per fold)")
logger.info("  - LOGS/03_kfold_summary.csv (overall summary)")

# Clean up any remaining memory
if DEVICE == 'cuda':
    torch.cuda.empty_cache()
