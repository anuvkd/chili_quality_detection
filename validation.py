from sklearn.model_selection import train_test_split
import os
import shutil

# Path to the directory containing the original dataset
dataset_dir = '/Users/nagesh/Documents/anu/chili_quality_detection/dataset'

# Directory for split datasets
base_dir = '/Users/nagesh/Documents/anu/chili_quality_detection/'
os.makedirs(base_dir, exist_ok=True)

# Subdirectories for training and validation data
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# List of chili grades 
grades = ['Grade1', 'Grade2', 'Grade3', 'Grade4']

# Split data into training and validation sets
for grade in grades:
    grade_dir = os.path.join(dataset_dir, grade)
    images = [img for img in os.listdir(grade_dir) if img.endswith('.JPG')]  # Adjust for your image format
    
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
    
    # Copy training images
    train_grade_dir = os.path.join(train_dir, grade)
    os.makedirs(train_grade_dir, exist_ok=True)
    for img in train_images:
        src = os.path.join(grade_dir, img)
        dst = os.path.join(train_grade_dir, img)
        shutil.copyfile(src, dst)
    
    # Copy validation images
    val_grade_dir = os.path.join(validation_dir, grade)
    os.makedirs(val_grade_dir, exist_ok=True)
    for img in val_images:
        src = os.path.join(grade_dir, img)
        dst = os.path.join(val_grade_dir, img)
        shutil.copyfile(src, dst)
