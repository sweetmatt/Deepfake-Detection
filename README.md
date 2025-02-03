# Deepfake-Detection
Our project is centered around deepfake detection methods and how we compare the effectiveness of each one using different datasets. After testing, we will make a UI application that allows users to test out the FACTOR method that produced the most effective results. Detection methods used in this project include FACTOR, LipForensics, and Self-Supervised Video Forensics by Audio-Visual Anamoly Detection.
# Project Set-Up And Access Requests
For this project, we completed all implementation through the OSC (Ohio Supercomputer Center) using the Jupyter notebooks with the following options: 
- Cluster: kubernetes
- Mode: Jupyter Notebook
- Node type: pitzer gpu
- GPUs: 4
- CUDA Version: cuda/11.8.0
- Number of cores: 8
- App Jupyter version: 3.1.18

The following access request forms were completed to use the datasets for the project (further instructions are included in the forms):
- FaceForensics++: https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform
- CelebDF-v2: https://docs.google.com/forms/d/e/1FAIpQLScoXint8ndZXyJi2Rcy4MvDHkkZLyBFKN43lTeyiG88wrG0rA/viewform
- FakeAVCeleb: https://docs.google.com/forms/d/e/1FAIpQLSfPDd3oV0auqmmWEgCSaTEQ6CGpFeB-ozQJ35x-B_0Xjd93bw/viewform

UI Development: For this project, we developed our UI using the Gradio Python Framework: https://www.gradio.app/guides/quickstart.
# Citations
We utilized this resource along with the GitHub repo created by Tal Reiss and their team. 
- Article: https://arxiv.org/abs/2311.01458 
- GitHub repo: https://github.com/talreiss/FACTOR
