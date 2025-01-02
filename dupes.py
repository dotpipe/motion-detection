import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def load_images_from_directory(directory):
    """
    Load all images from a directory.

    Args:
        directory (str): Path to the directory containing images.

    Returns:
        dict: A dictionary mapping filenames to their respective images.
    """
    images = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Supported image formats
            image = cv2.imread(file_path)
            if image is not None:
                images[filename] = image
    return images


def are_images_similar(image1, image2, similarity_threshold=0.2):
    """
    Compare two images for similarity using SSIM.

    Args:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.
        similarity_threshold (float): Threshold for similarity (0 to 1).

    Returns:
        bool: True if images are similar, False otherwise.
    """
    # Resize both images to the same size for comparison
    resized_image1 = cv2.resize(image1, (100, 100))
    resized_image2 = cv2.resize(image2, (100, 100))

    # Convert to grayscale
    gray_image1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    similarity, _ = ssim(gray_image1, gray_image2, full=True)

    return similarity >= similarity_threshold


def remove_duplicates(directory, similarity_threshold=0.45):
    """
    Remove duplicate images from a directory based on SSIM.

    Args:
        directory (str): Path to the directory containing images.
        similarity_threshold (float): Threshold for similarity (0 to 1).
    """
    # Load all images from the directory
    images = load_images_from_directory(directory)

    # Create a list to track which images are duplicates
    duplicates = set()

    # Compare each image with every other image
    filenames = list(images.keys())
    for i in range(len(filenames)):
        if filenames[i] in duplicates:
            continue  # Skip already marked duplicates

        for j in range(i + 1, len(filenames)):
            if filenames[j] in duplicates:
                continue  # Skip already marked duplicates

            # Compare the two images
            if are_images_similar(images[filenames[i]], images[filenames[j]], similarity_threshold):
                print(f"Duplicate found: {filenames[j]} is similar to {filenames[i]}")
                duplicates.add(filenames[j])

    # Remove duplicate images
    for duplicate in duplicates:
        duplicate_path = os.path.join(directory, duplicate)
        os.remove(duplicate_path)
        print(f"Removed duplicate: {duplicate}")


if __name__ == "__main__":
    # Directory containing cropped objects
    cropped_objects_dir = "cropped_objects"

    # Run the duplicate removal script
    remove_duplicates(cropped_objects_dir, similarity_threshold=0.15)
