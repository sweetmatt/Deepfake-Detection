# Depending on dataset structure, this code organizes the data by celebrity identifier for easier testing
import os
import shutil
import re

# Path for real videos
videos_dir = os.path.expanduser("~/FACTOR/face-forgery/FaceForensics_data/real_videos")

# Regular expression to extract actor ID
id_regex = re.compile(r"^(\d+)__.*\.mp4$")  # Matches digits at the beginning of the filename

# Iterate through video files
for filename in os.listdir(videos_dir):
    match = id_regex.match(filename)
    if match:
        actor_id = match.group(1)
        actor_dir = os.path.join(videos_dir, actor_id)  # Directory for the actor
        os.makedirs(actor_dir, exist_ok=True)  # Create directory if it doesn't exist

        source_path = os.path.join(videos_dir, filename)
        destination_path = os.path.join(actor_dir, filename)

        shutil.move(source_path, destination_path)  # Move the video file
        print(f"Moved {filename} to {actor_dir}")
    else:
        print(f"Filename {filename} does not match the expected pattern.")

# Path for fake videos
videos_dir = os.path.expanduser("~/FACTOR/face-forgery/FaceForensics_data/fake_videos")

# Regular expression to extract celeb ID (modified for the new filename pattern)
id_regex = re.compile(r"^(\d+)_.*\.mp4$")  # Matches digits at the beginning until the first underscore

# Iterate through video files
for filename in os.listdir(videos_dir):
    match = id_regex.match(filename)
    if match:
        celeb_id = match.group(1)
        celeb_dir = os.path.join(videos_dir, celeb_id)  # Directory for the celeb
        os.makedirs(celeb_dir, exist_ok=True)  # Create directory if it doesn't exist

        source_path = os.path.join(videos_dir, filename)
        destination_path = os.path.join(celeb_dir, filename)

        shutil.move(source_path, destination_path)  # Move the video file
        print(f"Moved {filename} to {celeb_dir}")
    else:
        print(f"Filename {filename} does not match the expected pattern: {filename}")  # More informative message

print("Video grouping complete.")
