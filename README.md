# Gram Staining Classification Project

## Overview

This project focuses on the classification of Gram-stained images using deep learning techniques. The workflow includes dataset generation, synthetic image creation, model architecture comparison, model selection, k-fold cross-validation, and final model evaluation. The project is organized into modular Python scripts, each responsible for a specific stage in the pipeline.

## Project Structure

```
├── 00_GenerateDataset.py                # Script to generate the initial dataset
├── 00_GenerateDataset_readme.txt        # Details about dataset generation
├── 00_GenerateSyntheticImages.py        # Script to create synthetic images
├── 00_GenerateSyntheticImages_readme.txt# Details about synthetic image generation
├── 01_model_arch_comparision.py         # Compare different model architectures
├── 01_model_arch_comparision_readme.txt # Details about architecture comparison
├── 02_model_selection.py                # Model selection process
├── 02_model_selection_readme.txt        # Details about model selection
├── 03_kfold_pipeline.py                 # K-fold cross-validation pipeline
├── 03_kfold_pipeline_readme.txt         # Details about k-fold pipeline
├── 04_final_model.py                    # Final model training and evaluation
├── 04_final_model_readme.txt            # Details about final model
├── requirements.txt                     # Python dependencies
├── LOGS/                                # Logs and results
│   ├── 01_model_comparision_results.csv
│   ├── 02_model_selection.csv
│   ├── 03_kfold_summary.csv
│   ├── 03_kfold_training_results.csv
│   ├── 04_final_model_results.csv
│   └── volo_d5_512_results_YYYYMMDD_HHMMSS/
│       ├── volo_d5_512_comprehensive_metrics.json
│       ├── volo_d5_512_confusion_matrix.csv
│       ├── volo_d5_512_overall_metrics.csv
│       ├── volo_d5_512_per_class_metrics.csv
│       └── volo_d5_512_training_summary.json
```

## Workflow

1. **Dataset Generation**

    - Run `00_GenerateDataset.py` to create the initial dataset from raw images.
    - Refer to `00_GenerateDataset_readme.txt` for details on input formats and parameters.

2. **Synthetic Image Generation**

    - Use `00_GenerateSyntheticImages.py` to augment the dataset with synthetic images.
    - Details in `00_GenerateSyntheticImages_readme.txt`.

3. **Model Architecture Comparison**

    - Execute `01_model_arch_comparision.py` to compare various deep learning models (e.g., ResNet, EfficientNet, VOLO).
    - Results are saved in `LOGS/01_model_comparision_results.csv`.
    - See `01_model_arch_comparision_readme.txt` for methodology.

4. **Model Selection**

    - Run `02_model_selection.py` to select the best-performing model based on comparison metrics.
    - Output in `LOGS/02_model_selection.csv`.
    - Details in `02_model_selection_readme.txt`.

5. **K-Fold Cross-Validation**

    - Use `03_kfold_pipeline.py` to perform k-fold cross-validation for robust evaluation.
    - Results in `LOGS/03_kfold_summary.csv` and `LOGS/03_kfold_training_results.csv`.
    - See `03_kfold_pipeline_readme.txt` for configuration.

6. **Final Model Training & Evaluation**
    - Train and evaluate the final model using `04_final_model.py`.
    - Comprehensive metrics, confusion matrix, and per-class results are stored in `LOGS/04_final_model_results.csv` and subfolders.
    - Details in `04_final_model_readme.txt`.

## Results & Logs

-   All experiment results, metrics, and logs are stored in the `LOGS/` directory.
-   Each major experiment creates a timestamped subfolder for reproducibility.
-   Metrics include accuracy, precision, recall, F1-score, confusion matrix, and per-class performance.

## Requirements

-   Python 3.8+
-   See `requirements.txt` for all dependencies (e.g., PyTorch, torchvision, scikit-learn, pandas, numpy, matplotlib).
-   Install dependencies with:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Clone the repository:
    ```bash
    git clone <repo_url>
    cd <repo_folder>
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Follow the workflow steps above, running each script in order.
4. Review logs and results in the `LOGS/` directory.

## Customization

-   Modify parameters in each script to adjust dataset paths, model hyperparameters, augmentation settings, and evaluation metrics.
-   Refer to the individual readme files for script-specific instructions.
