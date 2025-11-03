This script compares multiple deep learning model architectures for Gram staining image classification.
It trains and evaluates a list of models on the provided training and test datasets, logging performance metrics for each model.

How to Use:

1. Prepare your TRAIN and TEST folders, each with subfolders for the four classes (GNB, GNC, GPB, GPC).
2. Adjust script parameters if needed:
   - train_data_dir: Path to training images.
   - test_data_dir: Path to test images.
   - batch_size, epochs, model_names, etc.
3. Run the script:
   - The script will train and evaluate each model, saving results to LOGS/01_model_comparision_results.csv.
