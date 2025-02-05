# Extract frames from video directories, and then detect and align the faces within each frame
import os
import subprocess

# Paths
dataset_dir = os.path.expanduser("~/FACTOR/face-forgery/FaceForensics_data")  # Your dataset directory

# Constants
num_frames = 10

# Directories to process
directories_to_process = [
    "real_videos",  # Real videos
    "fake_videos"  # Fake videos
]

for relative_path in directories_to_process:
    videos_dir = os.path.join(dataset_dir, relative_path)

    if os.path.exists(videos_dir):
        for celeb_dir_name in os.listdir(videos_dir):  # Iterate through celeb/actor directories
            celeb_dir = os.path.join(videos_dir, celeb_dir_name)
            if os.path.isdir(celeb_dir):  # Check if it's a directory

                # Extract frames (on the celeb/actor directory level)
                extract_command = ["python", "extract_frames.py", "--input_root", celeb_dir, "--out_root", celeb_dir, "--num_frames", str(num_frames)]
                print("Executing Frame Extraction:", " ".join(extract_command))
                try:
                    subprocess.run(extract_command, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error extracting frames from {celeb_dir}: {e}")
                    print(f"Stderr: {e.stderr}")
                    print(f"Stdout: {e.stdout}")
                except FileNotFoundError:
                    print(f"extract_frames.py not found. Ensure the path is correct.")
                    exit(1)
    else:
        print(f"Directory {videos_dir} does not exist.")

for relative_path in directories_to_process:
    videos_dir = os.path.join(dataset_dir, relative_path)

    if os.path.exists(videos_dir):
        for celeb_dir_name in os.listdir(videos_dir):  # Iterate through celeb/actor directories
            celeb_dir = os.path.join(videos_dir, celeb_dir_name)
            if os.path.isdir(celeb_dir):  # Check if it's a directory
                for item in os.listdir(celeb_dir):
                    item_dir = os.path.join(celeb_dir, item)
                    if os.path.isdir(item_dir):
                        
                        frames_dir = item_dir #The item_path is the directory containing the frames

                        # Detect and Align (on the frames directory)
                        align_command = ["python", "detect_and_align.py", "--input_root", frames_dir, "--out_root", frames_dir]
                        print("Executing Detection and Alignment on {}:".format(item), " ".join(align_command)) #Print the directory name
                        try:
                            subprocess.run(align_command, check=True, capture_output=True, text=True)
                        except subprocess.CalledProcessError as e:
                            print(f"Error aligning frames from {frames_dir}: {e}")
                            print(f"Stderr: {e.stderr}")
                            print(f"Stdout: {e.stdout}")
                        except FileNotFoundError:
                            print(f"detect_and_align.py not found. Ensure the path is correct.")
                            exit(1)
    else:
        print(f"Directory {videos_dir} does not exist.")

print("Frame extraction and alignment complete.")
