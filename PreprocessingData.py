'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: disable oneDNN

import numpy as np
import shutil
import cv2
import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte  # Import for converting to uint8

# Set TensorFlow environment variables to reduce logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define input and output directories
input_dir = 'fer2013/train'
output_dir = 'fer2013-augmented/train'
target_count = {
    "happy": 7215,
    "sad": 9668,
    "neutral": 9888,
    "disgust": 5196,
    "fear": 8560,
    "surprise": 7071,
    "angry": 8358
}

# Initialize ImageDataGenerator for augmentation
data_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to compute LBP features
def compute_lbp(image):
    # Check if the image is in float format
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Ensure values are in the range [0, 1] before scaling to [0, 255]
        image = np.clip(image, 0, 1)  # Clip values to [0, 1]
        image = img_as_ubyte(image)  # Convert to 8-bit unsigned integer
    else:
        # Assume image is already uint8
        image = img_as_ubyte(image)

    # Parameters for LBP
    radius = 1
    n_points = 8 * radius
    # Compute LBP
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    return lbp

# Augment each emotion class
for emotion, count in target_count.items():
    total_count = 0

    emotion_dir = os.path.join(input_dir, emotion)
    output_emotion_dir = os.path.join(output_dir, emotion)
    os.makedirs(output_emotion_dir, exist_ok=True)  # Create output dir for the current emotion

    # List all images in the emotion directory
    images = os.listdir(emotion_dir)  # List images in the specific emotion directory

    for img_name in images:
        original_image_path = os.path.join(emotion_dir, img_name)
        # Extract the filename without extension
        filename, extension = os.path.splitext(img_name)
        # Construct the new filename
        new_filename = f"{emotion}_{total_count}{extension}"
        destination_image_path = os.path.join(output_emotion_dir, new_filename)
        shutil.copy2(original_image_path, destination_image_path)

        img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        lbp_features = compute_lbp(img_array.squeeze())  # Remove batch dimension
        lbp_features_path = os.path.join(f"lbp/{emotion}", f"{emotion}_{total_count}.npy")
        np.save(lbp_features_path, lbp_features)  # Save LBP features
        total_count += 1

    # Augment images until reaching the target count
    #total_count = len(images)

    # Ensure we have enough images to start augmentation
    if total_count == 0:
        print(f"No images found in {emotion_dir}. Skipping augmentation for this emotion.")
        continue

    current_img = 0
    while total_count < count:
        img_path = os.path.join(emotion_dir, images[current_img % len(images)])  # Loop through original images
        img = load_img(img_path, color_mode='grayscale')  # Load image in grayscale
        img_array = img_to_array(img)  # Convert to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Generate augmented images
        current_count = 0
        for batch in data_gen.flow(img_array, batch_size=1):
            aug_img_path = os.path.join(output_emotion_dir, f"{emotion}_{total_count}.jpg")
            save_img(aug_img_path, batch[0])  # Save the augmented image

            # Compute LBP features for the augmented image
            lbp_features = compute_lbp(batch[0].squeeze())  # Remove batch dimension
            lbp_features_path = os.path.join(f"lbp/{emotion}", f"{emotion}_{total_count}.npy")
            np.save(lbp_features_path, lbp_features)  # Save LBP features

            total_count += 1
            current_count += 1

            if current_count >= 2:  # Adjust this value based on how many augmentations you want
                break

        current_img += 1

print("Data augmentation with LBP extraction completed.")


# --------------------------------------------------------------------------------------------------------

# split augmented data into train, test, validation

# Set the directory paths
fer2013_dir = 'fer2013-augmented/train'
lbp_dir = 'lbp'
output_dirs = {
    'train': ('fer2013_split/train', 'lbp_split/train'),
    'validation': ('fer2013_split/validation', 'lbp_split/validation'),
    'test': ('fer2013_split/test', 'lbp_split/test')
}

# Create output directories for each split
for split in output_dirs.values():
    os.makedirs(split[0], exist_ok=True)  # FER2013 split
    os.makedirs(split[1], exist_ok=True)  # LBP split

# Set the split ratios
train_ratio, val_ratio, test_ratio = 0.2, 0.2, 0.6

# Process each class
for class_name in os.listdir(fer2013_dir):
    class_dir = os.path.join(fer2013_dir, class_name)
    lbp_class_dir = os.path.join(lbp_dir, class_name)

    # Skip if not a directory
    if not os.path.isdir(class_dir):
        continue

    # List all images in the class directory
    images = os.listdir(class_dir)
    random.shuffle(images)  # Shuffle for random splitting

    # Calculate split indices
    total_images = len(images)
    train_idx = int(total_images * train_ratio)
    val_idx = train_idx + int(total_images * val_ratio)

    # Split images for each set
    train_images = images[:train_idx]
    val_images = images[train_idx:val_idx]
    test_images = images[val_idx:]

    # Define a helper function to move images
    def move_images(image_list, split_name):
        fer_output_dir, lbp_output_dir = output_dirs[split_name]
        fer_class_output = os.path.join(fer_output_dir, class_name)
        lbp_class_output = os.path.join(lbp_output_dir, class_name)
        os.makedirs(fer_class_output, exist_ok=True)
        os.makedirs(lbp_class_output, exist_ok=True)

        for img_name in image_list:
            # Move image in fer2013
            src_img_path = os.path.join(class_dir, img_name)
            dst_img_path = os.path.join(fer_class_output, img_name)
            shutil.copy2(src_img_path, dst_img_path)

            # Move corresponding LBP feature in lbp
            lbp_name = img_name.replace('.jpg', '.npy')  # Assuming .npy for LBP files
            src_lbp_path = os.path.join(lbp_class_dir, lbp_name)
            dst_lbp_path = os.path.join(lbp_class_output, lbp_name)
            if os.path.exists(src_lbp_path):
                shutil.copy2(src_lbp_path, dst_lbp_path)

    # Move the images and their LBP features
    move_images(train_images, 'train')
    move_images(val_images, 'validation')
    move_images(test_images, 'test')

print("Splitting completed.")

#----------------------------------------------------------------------------------------
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: disable oneDNN

import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte
import cv2
from concurrent.futures import ThreadPoolExecutor

# Define directories
input_dir = 'fer2013/train'
output_dir = 'fer2013-augmented/train'
lbp_dir = 'lbp'

# Set target counts for each emotion
target_count = {
    "angry": 8358,
    "disgust": 5196,
    "fear": 8560,
    "happy": 7215,
    "neutral": 9888,
    "sad": 9668,
    "surprise": 7071
}
min_class_size = min(target_count.values())

# Calculate weights based on class sizes and the minimum class size
weights = {emotion: min_class_size / count for emotion, count in target_count.items()}
augmented_target_count = {emotion: int(count * weights[emotion]) for emotion, count in target_count.items()}

# Image Data Generator for augmentation
data_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(lbp_dir, exist_ok=True)

# Function to compute LBP features
def compute_lbp(image):
    image = img_as_ubyte(image)  # Converts to unsigned byte
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    return lbp

# Process images in parallel to copy and compute LBP features
def process_image(emotion, img_name, output_emotion_dir, lbp_dir):
    original_image_path = os.path.join(input_dir, emotion, img_name)
    img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Save original image to augmented directory
    destination_image_path = os.path.join(output_emotion_dir, img_name)
    shutil.copy2(original_image_path, destination_image_path)

    # Compute and save LBP features
    lbp_features = compute_lbp(img)
    lbp_features_path = os.path.join(lbp_dir, emotion, f"{emotion}_{img_name.split('.')[0]}.npy")
    np.save(lbp_features_path, lbp_features)

# Augment images up to the weighted target count for each class
for emotion, target in augmented_target_count.items():
    output_emotion_dir = os.path.join(output_dir, emotion)
    os.makedirs(output_emotion_dir, exist_ok=True)
    os.makedirs(os.path.join(lbp_dir, emotion), exist_ok=True)

    images = os.listdir(os.path.join(input_dir, emotion))
    total_count = len(images)

    # Copy original images and compute LBP features
    with ThreadPoolExecutor(max_workers=4) as executor:
        for img_name in images:
            executor.submit(process_image, emotion, img_name, output_emotion_dir, lbp_dir)

    # Perform augmentation up to the target count
    current_img = 0
    while total_count < target:
        img_path = os.path.join(input_dir, emotion, images[current_img % len(images)])
        img = load_img(img_path, color_mode='grayscale')
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        augment_count = 0
        for batch in data_gen.flow(img_array, batch_size=1):
            # Apply weighted average between original and augmented image
            aug_img = 0.7 * img_array[0] + 0.3 * batch[0]
            aug_img = aug_img.squeeze()  # Remove extra dimension

            # Normalize for img_as_ubyte compatibility
            aug_img_normalized = aug_img / 255.0

            aug_img_path = os.path.join(output_emotion_dir, f"{emotion}_{total_count}.jpg")
            save_img(aug_img_path, np.expand_dims(aug_img, axis=-1))

            # Compute and save LBP features of the normalized weighted image
            lbp_features = compute_lbp(aug_img_normalized)
            lbp_features_path = os.path.join(lbp_dir, emotion, f"{emotion}_{total_count}.npy")
            np.save(lbp_features_path, lbp_features)

            total_count += 1
            augment_count += 1

            if augment_count >= 2:  # Generate only 2 weighted-augmented images per original image
                break

        current_img += 1
        if current_img >= len(images):  # Loop through images
            current_img = 0

print("Weighted data augmentation and LBP extraction completed.")



