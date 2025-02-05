# Generate reference, test, and label files



# Generate reference file for all celebs
# Reference npy for all id's
import os
import numpy as np

# Paths
embeddings_dir = os.path.expanduser("~/FACTOR/face-forgery/FaceForensics_data/real_videos")

# Iterate through ID directories
for id_dir in os.listdir(embeddings_dir):

    id_path = os.path.join(embeddings_dir, id_dir)
    all_embeddings = []

    # Iterate through video directories within the ID directory
    for item in os.listdir(id_path):
        item_path = os.path.join(id_path, item)
        if os.path.isdir(item_path): #Check if item is a directory
            for file in os.listdir(item_path):
                if file.endswith(".npy"):
                    embedding_path = os.path.join(item_path, file)
                    embedding = np.load(embedding_path)
                    all_embeddings.append(embedding)

    # Combine all embeddings into a single array
    if all_embeddings:  # Check if any embeddings were found
        combined_embedding = np.concatenate(all_embeddings)
        # Save the combined embedding as reference features npy
        reference_path = os.path.join(id_path, f"{id_dir}_reference.npy")
        np.save(reference_path, combined_embedding)
        print(f"Reference features saved for {id_dir}: {reference_path}")
    else:
        print(f"No .npy files found in {id_dir}. Skipping reference feature generation.")

print("Reference feature generation complete.")





# Generate test and label file for each celebrity video
# Define base test directory
test_dir = "FACTOR/face-forgery/FaceForensics_data/test"

def process_video_dir(video_dir_path, label, video_data):
    """Processes all video frame data folders inside a given directory, extracts .npy embeddings, and averages them."""
    for video_name in os.listdir(video_dir_path):
        video_path = os.path.join(video_dir_path, video_name)
        if not os.path.isdir(video_path):  # Skip non-directory files
            continue
        
        features = []
        for file in os.listdir(video_path):
            if file.endswith(".npy"):
                embedding = np.load(os.path.join(video_path, file))
                features.append(embedding)

        if features:
            features_array = np.squeeze(np.array(features))  # Remove any extra dimensions
            print(f"Shape of features before averaging for {video_name}:", features_array.shape)
            average_feature = np.mean(features_array, axis=0)  # Compute average feature embedding
            print(f"Shape of average_feature after averaging for {video_name}:", average_feature.shape)
            video_data[video_name] = (average_feature, label)
        else:
            print(f"No features found for {video_path}")

# Iterate through each celebrity directory in the test set
for celeb_id in os.listdir(test_dir):
    celeb_path = os.path.join(test_dir, celeb_id)
    if not os.path.isdir(celeb_path):  # Skip non-directory files
        continue

    real_dir = os.path.join(celeb_path, "real")
    fake_dir = os.path.join(celeb_path, "fake")

    video_data = {}  # Dictionary to store video features and labels

    # Process real and fake videos
    if os.path.exists(real_dir):
        process_video_dir(real_dir, 0, video_data)
    if os.path.exists(fake_dir):
        process_video_dir(fake_dir, 1, video_data)

    # Convert to arrays
    test_embeddings = np.array([data[0] for data in video_data.values()])
    labels = np.array([data[1] for data in video_data.values()])

    # Save the embeddings and labels in the corresponding celebrity directory
    np.save(os.path.join(celeb_path, f"id{celeb_id}_test.npy"), test_embeddings)
    np.save(os.path.join(celeb_path, f"id{celeb_id}_labels.npy"), labels)

    print(f"Saved embeddings for {celeb_id} in {celeb_path}")
    print("Test embeddings shape:", test_embeddings.shape)
    print("Labels shape:", labels.shape)
