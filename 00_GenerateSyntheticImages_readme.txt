This script creates synthetic images using the MixUp technique.
It blends pairs of images from a specified input folder to generate new,
augmented images. The output images are resized, randomly flipped, color-jittered, and saved to a chosen output directory. You can control the number of synthetic images, the output size, and the mixing strength.

How to Use:

1. Place your source images in a folder (e.g., GNC).

2. Adjust the script’s parameters if needed:
    input_folder_path: Path to your source images.
    output_directory: Where to save synthetic images.
    image_rescale_size: Output image size (default (300, 300)).
    n: Number of synthetic images to generate.

3. Run the script:
    The synthetic images will be saved in the specified output directory.