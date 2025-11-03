This script evaluates different variants of a selected model architecture for Gram staining classification.
It trains and compares several model variants on the combined dataset, logging performance metrics for each.

How to Use:

1. Ensure TRAIN and TEST folders are prepared as described in previous steps.
2. Adjust script parameters if needed:
   - model_list: List of model variants to compare.
   - train_data_dir, test_data_dir, epochs, etc.
3. Run the script:
   - The script will train and evaluate each model variant, saving results to LOGS/02_model_selection.csv.
