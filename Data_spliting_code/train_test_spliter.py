import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(dataset_dir, output_dir, test_size=0.1, random_state=42):
    """
    Split a dataset into train and test sets, maintaining class-wise subdirectories.

    Args:
        dataset_dir (str): Path to the dataset directory (e.g., "dataset/").
        output_dir (str): Path to the output directory for split data (e.g., "split_dataset/").
        test_size (float): Proportion of the dataset to include in the test split (default: 0.2).
        random_state (int): Seed for reproducibility (default: 42).
    """
    # Define output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate through each class directory
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue  # Skip non-directory files

        # Get all image file paths in the class directory
        image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Split into train and test sets
        train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=random_state)

        # Create corresponding class subdirectories in train and test folders
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Move files to respective directories
        for file in train_files:
            shutil.copy(file, train_class_dir)
        for file in test_files:
            shutil.copy(file, test_class_dir)

    print(f"Dataset split completed. Train and test data saved in '{output_dir}'.")

# Example usage
dataset_path = "Roads_and_regions_data_23_jan_2025"  # Input dataset directory
output_path = "Roads_and_regions_dataset"  # Output directory for train/test split
split_dataset(dataset_path, output_path, test_size=0.1)
