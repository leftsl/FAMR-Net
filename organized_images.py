import shutil
import pandas as pd
import os
from tqdm import tqdm
import time

# Read the CSV file, assuming the CSV file is comma-separated
metadata_file = 'IU_XRAY_metadata.csv'  # Make sure the filename and path are correct
df = pd.read_csv(metadata_file)

# Original and target paths for the images
original_path = 'IU_XRAY/images'
target_path = 'IU_XRAY/organized_images'

# Ensure the target path exists, if not, create it
if not os.path.exists(target_path):
    os.makedirs(target_path)

# Iterate over each row in the CSV file
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # Get the image name and label
    image_name = row['ImageID']  # Assuming the CSV file has a column named 'ImageID'
    label = row['Finding Label']  # Assuming the CSV file has a column named 'Finding Label'
    
    # Build the full path for the original image
    original_image_path = os.path.join(original_path, image_name + '.png')
    
    # Check if the file exists, if not, skip
    if not os.path.exists(original_image_path):
        print(f"File does not exist: {original_image_path}")
        continue
    
    # Ensure the label folder exists in the target path, if not, create it
    label_folder = os.path.join(target_path, label)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    
    # Build the full path for the target image
    target_image_path = os.path.join(label_folder, image_name + '.png')
    
    # Move the image
    shutil.copy(original_image_path, target_image_path)

print("Image classification completed.")
