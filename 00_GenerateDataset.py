"""
Copy files from SET01 and SET 02 over to a train set folder
Picks 50 random images from each folder to make a test set
"""


import os
import random
import shutil

# Define the source and destination folders
src1 = '../DATA/SET01'
src2 = '../DATA/SET02'
dest = '../DATA/train_test_set'


# -----------------------------------------------------------------
# Class List
classes = ['GNB', 'GNC', 'GPB', 'GPC']

# Create the destination folder if it doesn't exist
if not os.path.exists(dest):
    os.makedirs(dest)
    os.makedirs(f'{dest}/TRAIN')
    os.makedirs(f'{dest}/TEST')
    for i in classes:
        os.makedirs(f'{dest}/TRAIN/{i}')
        os.makedirs(f'{dest}/TEST/{i}')

# Copy files from SET01 and SET 02 over to a train set folder
for i in classes:
    print(f"Copying {i}")
    files = os.listdir(f'{src1}/{i}')
    for j in range(len(files)):
        shutil.copy(f'{src1}/{i}/{files[j]}', f'{dest}/TRAIN/{i}/{files[j]}')

    files = os.listdir(f'{src2}/{i}')
    for j in range(len(files)):
        shutil.copy(f'{src2}/{i}/{files[j]}', f'{dest}/TRAIN/{i}/{files[j]}')

print("Created train set")

# Pick 30 random images from each folder to make a test set
for i in classes:
    files = os.listdir(f'{dest}/TRAIN/{i}')
    random.shuffle(files)
    for j in range(30):
        shutil.move(f'{dest}/TRAIN/{i}/{files[j]}', f'{dest}/TEST/{i}/{files[j]}')

print("Created test set")

