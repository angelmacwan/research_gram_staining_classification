'''
This script generates synthetic images using a MixUp approach by blending pairs of images.
It takes the input folder path, output directory, image rescale size, and number of synthetic images as input.
'''

import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms

def mixup_images(input_folder_path, output_dir,image_rescale_size=(300, 300),  n=100, alpha=0.4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read all the image paths, ignoring files starting with dot
    image_paths = [os.path.join(input_folder_path, f) for f in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, f)) and not f.startswith('.')]
    if len(image_paths) < 2:
        return "Error: There must be at least two images in the input folder."

    # Transform to ensure consistent sizes and add variability
    transform = transforms.Compose([
        transforms.Resize(image_rescale_size), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), 
        transforms.ToTensor()
    ])
    to_pil = transforms.ToPILImage()
    
    # generate list of all possible pairs
    image_pairs = []
    for i in range(len(image_paths)):
        for j in range(i+1, len(image_paths)):
            image_pairs.append([image_paths[i], image_paths[j]])
    
    random.shuffle(image_pairs)
    
    for i in range(n):
        if len(image_pairs) == 0:
            print("ALL IMAGE PAIRS USED")
            break

        img_path1, img_path2 = image_pairs.pop()

        # Load and transform the images
        img1 = transform(Image.open(img_path1).convert("RGB"))
        img2 = transform(Image.open(img_path2).convert("RGB"))

        # Generate a random mixing coefficient lambda from the Beta distribution
        lam = np.random.beta(alpha, alpha)

        # Create a mixed image using the MixUp formula
        mixed_image = lam * img1 + (1 - lam) * img2

        # Convert the tensor back to a PIL Image
        mixed_image_pil = to_pil(mixed_image.clamp(0, 1))  # Ensure pixel values are within [0, 1]

        # Save the mixed image
        output_path = os.path.join(output_dir, f"mixed_image_{i + 1}.jpg")
        mixed_image_pil.save(output_path)

    print(f"Images saved to {output_dir}")


if __name__ == "__main__":
    input_folder_path = '../DATA/SET01/GNC'
    output_directory = "./mix_gnc"
    mixup_images(input_folder_path, output_directory,image_rescale_size=(300, 300), n=80)
