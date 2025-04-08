import os
import shutil
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm

def organize_dataset(source_dir, dest_dir, validation_split=0.2):
    """
    Organize the dataset into train and validation sets.
    
    Args:
        source_dir: Directory containing the original dataset
        dest_dir: Directory where to create train and validation splits
        validation_split: Fraction of data to use for validation
    """
    # Create destination directories
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all disease categories (folders in the dataset)
    categories = [d for d in os.listdir(source_dir) 
                 if os.path.isdir(os.path.join(source_dir, d))
                 and d != 'Background_without_leaves']  # Exclude background images
    
    print(f"Found {len(categories)} disease categories")
    
    for category in tqdm(categories, desc="Processing categories"):
        print(f"\nProcessing {category}...")
        
        # Create category directories in train and validation
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        
        # Get all images in the category
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(source_dir, category, ext)))
        
        if not image_files:
            print(f"Warning: No images found in {category}")
            continue
        
        # Split into train and validation
        train_files, val_files = train_test_split(
            image_files, 
            test_size=validation_split,
            random_state=42
        )
        
        # Copy files to respective directories
        for file in tqdm(train_files, desc="Copying training files"):
            shutil.copy2(
                file,
                os.path.join(train_dir, category, os.path.basename(file))
            )
        
        for file in tqdm(val_files, desc="Copying validation files"):
            shutil.copy2(
                file,
                os.path.join(val_dir, category, os.path.basename(file))
            )
        
        print(f"  {len(train_files)} training images")
        print(f"  {len(val_files)} validation images")

if __name__ == '__main__':
    # Update these paths according to your setup
    source_directory = os.path.join('data', 'PlantVillage', 'Plant_leave_diseases_dataset_with_augmentation')
    destination_directory = os.path.join('data', 'organized')
    
    print("Starting dataset organization...")
    organize_dataset(source_directory, destination_directory)
    print("\nDataset organization complete!")
    
    # Update the training script paths
    with open('train_model.py', 'r') as f:
        content = f.read()
    
    content = content.replace(
        "train_dir = os.path.join('..', 'data', 'organized', 'train')",
        "train_dir = os.path.join('data', 'organized', 'train')"
    )
    content = content.replace(
        "validation_dir = os.path.join('..', 'data', 'organized', 'validation')",
        "validation_dir = os.path.join('data', 'organized', 'validation')"
    )
    
    with open('train_model.py', 'w') as f:
        f.write(content)
    
    print("\nUpdated training script paths!") 