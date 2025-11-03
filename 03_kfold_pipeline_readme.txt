This script performs k-fold cross-validation for Gram staining image classification using a selected model architecture.
It splits the training data into k folds, trains and validates the model on each fold, and logs detailed metrics.

How to Use:

1. Prepare your TRAIN folder with subfolders for each class (GNB, GNC, GPB, GPC).
2. Adjust script parameters if needed:
   - model_name, k_folds, num_epochs, etc.
3. Run the script:
   - The script will perform k-fold training and validation, saving results to LOGS/03_kfold_training_results.csv and LOGS/03_kfold_summary.csv.
