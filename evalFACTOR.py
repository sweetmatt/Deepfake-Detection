# Evaluates FACTOR's accuracy for each celebrity

import os
import subprocess
import re

# Paths (modify as needed)
embeddings_dir = os.path.expanduser("~/FACTOR/face-forgery/FaceForensics_data/test")
reference_dir = os.path.expanduser("~/FACTOR/face-forgery/FaceForensics_data/real_videos")  # Assuming reference embeddings are in id folders

ap_scores = []
auc_scores = []

# Process all test directories
for celeb in os.listdir(embeddings_dir):
  celeb_dir = os.path.join(embeddings_dir, celeb)
  if os.path.isdir(celeb_dir):
    id_number = celeb

    # Construct paths based on ID directory name
    reference_path = os.path.join(reference_dir, id_number, f"{id_number}_reference.npy")
    test_path = os.path.join(celeb_dir, f"id{id_number}_test.npy")
    labels_path = os.path.join(celeb_dir, f"id{id_number}_labels.npy")

    # Check if necessary files exist
    if not (os.path.exists(reference_path) and os.path.exists(test_path) and os.path.exists(labels_path)):
      print(f"Skipping {celeb_dir}: Missing required files.")
      continue

    # Evaluate accuracy using the external script (one-liner)
    command = f"python eval.py --real_root {reference_path} --test_root {test_path} --labels_root {labels_path}"
    result = subprocess.run(command.split(), capture_output=True, text=True)
    
    # Extract AP and AUC using regular expressions
    output = result.stdout
    ap_match = re.search(r"AP:\s*([\d.]+)", output)
    auc_match = re.search(r"AUC:\s*([\d.]+)", output)

    if ap_match and auc_match:
        ap = float(ap_match.group(1))
        auc = float(auc_match.group(1))
        ap_scores.append(ap)
        auc_scores.append(auc)
        print(f"Accuracy for {celeb_dir}: AP: {ap}, AUC: {auc}")
    else:
        print(f"Could not parse AP or AUC from eval.py output for {celeb_dir}. Output was: {output}")

if ap_scores and auc_scores:
    avg_ap = sum(ap_scores) / len(ap_scores)
    avg_auc = sum(auc_scores) / len(auc_scores)
    print(f"\nAverage AP across all IDs: {avg_ap}")
    print(f"Average AUC across all IDs: {avg_auc}")
else:
    print("No AP or AUC scores were collected. Check eval.py outputs.")

print("Accuracy evaluation complete.")
