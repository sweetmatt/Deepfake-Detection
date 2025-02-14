import gradio as gr
import os
import shutil
import subprocess
import numpy as np
import faiss
from pathlib import Path

# Define persistent directories
BASE_DIR = os.path.expanduser("/users/PMIU0184/sweetmatt/DeepfakeApp")
REAL_VIDEOS_DIR = os.path.join(BASE_DIR, "real_videos")
TEST_VIDEOS_DIR = os.path.join(BASE_DIR, "test_videos")
FRAMES_DIR = os.path.join(BASE_DIR, "frames")

# Create directories if they do not exist
os.makedirs(REAL_VIDEOS_DIR, exist_ok=True)
os.makedirs(TEST_VIDEOS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# Paths to external scripts
EXTRACT_FRAMES_SCRIPT = os.path.expanduser("~/FACTOR/FaceX-Zoo/face_sdk/extract_frames.py")
DETECT_ALIGN_SCRIPT = os.path.expanduser("~/FACTOR/FaceX-Zoo/face_sdk/detect_and_align.py")
EXTRACT_FEATURES_SCRIPT = os.path.expanduser("~/FACTOR/FaceX-Zoo/test_protocol/extract_feature.py")

def extract_frames_for_all_videos(video_list, category):
    """
    Extracts frames for all videos and stores them in /DeepfakeApp/frames/real or /DeepfakeApp/frames/fake.
    """
    if not video_list:
        return []

    video_dir = REAL_VIDEOS_DIR if category == "Real Videos" else TEST_VIDEOS_DIR
    frame_subdir = "real" if category == "Real Videos" else "fake"
    frame_base_dir = os.path.join(FRAMES_DIR, frame_subdir)  # /DeepfakeApp/frames/real or /DeepfakeApp/frames/fake

    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(frame_base_dir, exist_ok=True)

    frame_dirs = []

    for video_path in video_list:
        video_name = Path(video_path).stem
        stored_video_path = os.path.join(video_dir, Path(video_path).name)
        shutil.move(video_path, stored_video_path)

        # Store frames in /DeepfakeApp/frames/real or /fake with the video name
        frame_dir = os.path.join(frame_base_dir, video_name)
        os.makedirs(frame_dir, exist_ok=True)

        frame_dirs.append(frame_dir)

    # Extract frames for all videos
    os.chdir("/users/PMIU0184/sweetmatt/FACTOR/FaceX-Zoo/face_sdk")
    extract_frames_command = [
        "python", EXTRACT_FRAMES_SCRIPT,
        "--input_root", video_dir,
        "--out_root", frame_base_dir,  # Output frames to /DeepfakeApp/frames/real or /fake
    ]
    print("Executing Frame Extraction:", " ".join(extract_frames_command))
    try:
        subprocess.run(extract_frames_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
    except FileNotFoundError:
        print(f"Error: {EXTRACT_FRAMES_SCRIPT} not found.")

    # Face alignment for each frame directory
    for frame_dir in frame_dirs:
        align_command = [
            "python", DETECT_ALIGN_SCRIPT,
            "--input_root", frame_dir,  # Specific video frame directory
            "--out_root", frame_dir  # Output aligned frames in the same directory
        ]
        print(f"Executing Alignment for {frame_dir}:", " ".join(align_command))
        try:
            subprocess.run(align_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during alignment: {e}")
            print(f"Stderr: {e.stderr}")
            print(f"Stdout: {e.stdout}")
        except FileNotFoundError:
            print(f"Error: {DETECT_ALIGN_SCRIPT} not found.")

    return frame_dirs



def extract_features_for_each_video(frame_dirs):
    """
    Extracts features for each video individually after alignment.
    """
    feature_files = []

    os.chdir("/users/PMIU0184/sweetmatt/FACTOR/FaceX-Zoo/test_protocol")
    for frame_dir in frame_dirs:

        # Extract features (reference.npy will be generated inside `aligned_frame_dir`)
        extract_features_command = [
            "python", EXTRACT_FEATURES_SCRIPT,
            "--input_root", frame_dir  # Directory containing aligned frames
        ]
        print(f"Executing Feature Extraction for {frame_dir}:", " ".join(extract_features_command))
        try:
            subprocess.run(extract_features_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting features: {e}")
            print(f"Stderr: {e.stderr}")
            print(f"Stdout: {e.stdout}")
        except FileNotFoundError:
            print(f"Error: {EXTRACT_FEATURES_SCRIPT} not found.")

        # Check if reference.npy was generated
        feature_path = os.path.join(frame_dir, "features.npy")
        if os.path.exists(feature_path):
            feature_files.append(feature_path)

    return feature_files

def generate_reference_feature(real_feature_files):
    """
    Combines all real video `reference.npy` files into a single dataset.
    """
    if not real_feature_files:
        return None

    all_embeddings = [np.load(f) for f in real_feature_files]
    combined_embedding = np.concatenate(all_embeddings, axis=0)
    reference_path = os.path.join(REAL_VIDEOS_DIR, "reference.npy")
    np.save(reference_path, combined_embedding)
    return reference_path

def compare_features(reference_feature_file, test_feature_files):
    """
    Compares test video features against the reference feature set using FAISS.
    """
    if not reference_feature_file or not test_feature_files:
        return "Error: Missing reference or test features."

    reference_features = np.load(reference_feature_file).astype(np.float32)
    test_features = np.vstack([np.load(f).astype(np.float32) for f in test_feature_files])

    # FAISS Index for similarity search
    index = faiss.IndexFlatL2(reference_features.shape[1])
    index.add(reference_features)

    k_value = 1
    D, _ = index.search(test_features, k_value)
    distances = np.sum(D, axis=1)

    avg_dist = np.mean(distances)
    
    print(avg_dist)
    
    # Thresholding based on distance
    threshold = 50
    prediction = "FAKE" if avg_dist > threshold else "REAL"

    return f"Deepfake analysis completed. Prediction:\n{prediction}"

def process_videos(real_video_list, fake_video_list):
    """
    Handles video uploads, extracts features, generates reference, and compares test videos.
    """
    # Step 1: Extract frames for all uploaded videos
    frame_dirs_real = extract_frames_for_all_videos(real_video_list, "Real Videos")
    frame_dirs_fake = extract_frames_for_all_videos(fake_video_list, "Test Videos")

    # Step 3: Extract features for each video
    real_features = extract_features_for_each_video(frame_dirs_real)
    fake_features = extract_features_for_each_video(frame_dirs_fake)

    # Step 4: Generate reference features and compare test features
    reference_feature_file = generate_reference_feature(real_features)
    return compare_features(reference_feature_file, fake_features)

# Gradio Interface
iface = gr.Interface(
    fn=process_videos,
    inputs=[
        gr.Files(file_types=[".mp4", ".avi", ".mov"], type="filepath", label="Upload Real Videos", interactive=True),
        gr.Files(file_types=[".mp4", ".avi", ".mov"], type="filepath", label="Upload Test Videos", interactive=True),
    ],
    outputs="text",
    title="Deepfake Detection Tool",
    description="Upload real and test videos. A reference set is created from real videos, and test videos are analyzed against it.",
)

# Launch the Gradio app
iface.launch(share=True)
