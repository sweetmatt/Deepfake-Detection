# Move celebrity feature files to appropriate test directories

import os
import shutil
import random

# Create test subdirectories for each actor and move 25% of real videos to the test set

# Define source and destination directories
source_dir = "FACTOR/face-forgery/FaceForensics_data/real_videos"
test_dir = "FACTOR/face-forgery/FaceForensics_data/test"

# Ensure the destination directory exists
os.makedirs(test_dir, exist_ok=True)

# Get all subdirectories in the source directory
celeb_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

# Iterate through found celebrity directories
for celeb in celeb_dirs:
    celeb_source_path = os.path.join(source_dir, celeb)
    celeb_test_path = os.path.join(test_dir, celeb)

    os.makedirs(celeb_test_path, exist_ok=True)
    print(f"Created directory: {celeb_test_path}")

    # Create "real" and "fake" subdirectories under the test directory
    real_dest_path = os.path.join(celeb_test_path, "real")
    fake_dest_path = os.path.join(celeb_test_path, "fake")

    os.makedirs(real_dest_path, exist_ok=True)
    os.makedirs(fake_dest_path, exist_ok=True)
    print(f"Created directory: {real_dest_path}")
    print(f"Created directory: {fake_dest_path}")

    # Get all frame data directories inside the celeb's source folder
    frame_dirs = [d for d in os.listdir(celeb_source_path) if os.path.isdir(os.path.join(celeb_source_path, d))]

    # Randomly select 1/4 of these directories for moving
    num_to_move = max(1, len(frame_dirs) // 4)  # Ensure at least 1 folder is moved
    selected_dirs = random.sample(frame_dirs, num_to_move)

    # Move the selected directories to the "real" folder in the test directory
    for frame_dir in selected_dirs:
        src = os.path.join(celeb_source_path, frame_dir)
        dest = os.path.join(real_dest_path, frame_dir)
        shutil.move(src, dest)
        print(f"Moved {src} -> {dest}")


# Move all deepfake videos to test directory

# Define source of fake videos
fake_source_dir = "FACTOR/face-forgery/FaceForensics_data/fake_videos"

# Get all celebrity directories in the fake_videos directory
celeb_dirs = [d for d in os.listdir(fake_source_dir) if os.path.isdir(os.path.join(fake_source_dir, d))]

# Iterate through each celebrity directory
for celeb in celeb_dirs:
    celeb_source_path = os.path.join(fake_source_dir, celeb)
    celeb_test_path = os.path.join(test_dir, celeb, "fake")  # Destination folder

    # Ensure the destination "fake" folder exists
    os.makedirs(celeb_test_path, exist_ok=True)

    # Get all frame data directories inside the celeb's fake video folder
    frame_dirs = [d for d in os.listdir(celeb_source_path) if os.path.isdir(os.path.join(celeb_source_path, d))]

    # Move each frame data directory to the "fake" subdirectory in test/
    for frame_dir in frame_dirs:
        src = os.path.join(celeb_source_path, frame_dir)
        dest = os.path.join(celeb_test_path, frame_dir)
        shutil.move(src, dest)
        print(f"Moved {src} -> {dest}")
