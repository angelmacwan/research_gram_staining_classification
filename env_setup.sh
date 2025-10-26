echo "ENV_SETUP"
pip install torch torchvision pandas numpy scikit-image scikit-learn scipy timm

echo "Downloading Dataset"
wget https://storage.googleapis.com/gram_staining_dataset/Archive.zip
unzip Archive.zip

echo "DONE"

