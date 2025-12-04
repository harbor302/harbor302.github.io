---
permalink: /
title: "Harbor (Haobo) Liu"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

**Seeking MLE Intern for Summer 2026 · MSCS @ NYU Courant ’27 · Ex-Baidu ML Engineer Intern**

**Contact:** +1 (201) 589-9209 · hl6597@nyu.edu · [LinkedIn](https://www.linkedin.com/in/harbor-liu) · [GitHub](https://github.com/harbor302)

---

## Summary

Machine learning engineer with 3+ internships delivering anti-fraud systems, multimodal research, and robotics perception stacks. I ship computer-vision pipelines end-to-end—from data collection, model training, deployment on Kubernetes/Hadoop/embedded boards, to evaluation with F1/AUC/ROC. Currently preparing to join NYU Courant’s MSCS program (’27) and actively seeking a Summer 2026 MLE internship.

## Experience Highlights

### Machine Learning Engineer Intern · Baidu, Inc. · Beijing, China (Nov 2023 – Nov 2024)
- Built duplicate-face detection with FFmpeg sampling, MTCNN alignment, CosFace embeddings, and ANN clustering to automate 1.6k+ takedowns/day.
- Delivered an anti-fraud platform using few-shot/self-training models (SMOTE, LOF, stacking, TrAdaBoost) that lifted recall from 74% → 92% and precision from 75% → 96%.
- Operationalized GraphSAGE + Louvain graph learning for fraud communities plus ASR/OCR-based risk scoring with XGBoost/GBDT/RF models on Kubernetes.

### Machine Learning Engineer Intern · China National Petroleum Corporation · Beijing, China (Jun 2024 – Jul 2024)
- Automated SAM-based labeling, OpenCV preprocessing, and Dockerized Faster R-CNN training; boosted mAP@[0.5:0.95] to 42.4 on 12k photovoltaic images.
- Parallelized training across Hadoop to cut epoch time by 35% and harden the anomaly detection pipeline for field deployment.

### Machine Learning Engineer Intern · Ragine Technology · Xi’an, China (Jul 2023 – Sep 2023)
- Scaled datasets from 1k → 15k via mosaic and synthetic occlusion, re-clustered YOLOv8 anchors, and tuned NMS/IoU for IR targets.
- Combined U-Net segmentation with OSTrack tracking and deployed the vision stack on embedded edge devices for real-time drone imagery.

## Research & Leadership

### CV Group Lead · RoboMaster & Robocon Champion · CUPB Robotics Team (Sep 2022 – Jun 2025)
- Integrated RGB-D + LiDAR fusion, pose estimation, and depth-based yaw control using YOLOv5 + Kalman filtering on ROS2/Linux with serial I/O.
- Designed TD-learning decision policies with ε-greedy exploration, early pruning, and hash-based deduplication; deployed OpenVINO pipelines on Ubuntu NUCs.
- Built Webots simulations for perception-control testing and automated deployment via Shell scripts.

### Research Assistant · CUPB (Jan 2025 – May 2025)
- Fused ResNet-50 vision and RoBERTa text streams via Transformer cross-attention to beat the MMBT baseline by 8% on multimodal sentiment tasks.
- Fine-tuned Stable Diffusion with LoRA (rank 8) to enable emotion-conditioned image synthesis for internal studies.

### Research Assistant · Xidian University (Jul 2023 – Sep 2023)
- Led YOLOv8 anchor re-clustering, U-Net segmentation, and OSTrack tracking for embedded inference in resource-constrained deployments.

## Education

- **New York University – Courant Institute of Mathematical Sciences** · M.S. Computer Science · Sept 2025 – May 2027 (expected) · New York, NY  
- **China University of Petroleum (Beijing)** · B.Eng. Artificial Intelligence · Sept 2021 – Jun 2025 · Beijing, China

## Skills Snapshot

- **Languages:** Python (Pandas, NumPy, scikit-learn, Matplotlib), C/C++, Bash, SQL, Racket, HTML/CSS, CUDA  
- **Developer Tools:** GitHub, GitLab, MLflow, Docker, Kubernetes, Flask, AWS, Azure, GCP, Hadoop, Spark, Xmind  
- **Frameworks & Models:** PyTorch, TensorFlow, Keras, Hugging Face, OpenCV, YOLO (v5/v8), U-Net, SAM, GraphSAGE, Transformers, Stable Diffusion, LLaMA, GPT

## Publication

- **CaSTGCN: Deep Learning Method for Information Cascade Prediction** · IEEE CAC 2025  
  Temporal convolution + spatio-temporal GCN with dual-branch attention, Optuna-driven tuning (400 trials), and training refinements (MultiStepLR, dropout, warmup, gradient clipping) that reduced MSLE by 15% versus baselines.
