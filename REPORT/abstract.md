# Abstract

## Objective

To develop and evaluate deep learning models for automated classification of gram-stained bacterial images into four distinct morphological classes (Gram-negative bacilli, Gram-negative cocci, Gram-positive bacilli, and Gram-positive cocci).

## Design

Systematic comparative study using cross-sectional image classification with k-fold cross-validation to evaluate model robustness and generalization performance.

## Subjects, Participants, and Controls

The study utilized gram-stained bacterial image datasets (SET01 and SET02) organized into four classes: GNB (Gram-negative bacilli), GNC (Gram-negative cocci), GPB (Gram-positive bacilli), and GPC (Gram-positive cocci). Test set comprised 30 images per class; remaining images formed the training set.

## Methods, Intervention, and Testing

A multi-stage machine learning pipeline was implemented: (1) dataset organization and preprocessing with random train/test split (30 images per class in test set), (2) comparison of 24 different deep learning architectures (including ResNet, EfficientNet, Vision Transformer, and VOLO variants), (3) model variant selection from best-performing architecture, (4) 5-fold cross-validation for robust evaluation, and (5) comprehensive final model assessment with extensive data augmentation techniques during training (horizontal/vertical flip, rotation, color jittering, Gaussian blur, and random erasing).

## Main Outcome Measures

Primary outcome measures included classification accuracy, precision, recall, weighted F1-score, sensitivity, specificity, and per-class performance metrics. Confusion matrix analysis was performed for detailed classification performance assessment.

## Results

Architecture comparison identified VOLO variants as superior performers. The best model (VOLO-D3 448) achieved 89.5% accuracy with F1-score of 0.894 on the test set. K-fold cross-validation yielded a mean accuracy of 88.85% (±0.96%) with consistent F1-scores across folds (0.889 ± 0.009). Final model evaluation demonstrated balanced per-class performance with F1-scores ranging from 0.843 to 0.900, indicating effective discrimination across all bacterial morphology classes.

## Conclusions

Vision transformer-based architectures, particularly VOLO variants, demonstrate superior performance for automated gram-stained bacterial image classification. The consistent performance across k-fold validation and excellent per-class metrics suggest the developed model is robust and suitable for automated microbial morphology classification in clinical settings.
