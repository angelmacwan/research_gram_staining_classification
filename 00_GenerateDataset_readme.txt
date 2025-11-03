This script organizes image data for Gram staining classification.
It collects images from two source folders (SET01 and SET02), merges them into a training set, and then randomly selects 30 images per class to create a test set.
The images are sorted into four classes: GNB, GNC, GPB, and GPC.

How to Use:

Ensure your source folders (SET01 and SET02) are structured with subfolders for each class (GNB, GNC, GPB, GPC), each containing images.
Run the script:
The script will create a new folder (train_test_set) with TRAIN and TEST subfolders, each containing class-specific folders with the appropriate images.