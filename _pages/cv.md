---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

## Harbor (Haobo) Liu

- **Email:** hl6597@nyu.edu · **Phone:** +1 (201) 589-9209  
- **LinkedIn:** [linkedin.com/in/harbor-liu](https://www.linkedin.com/in/harbor-liu) · **GitHub:** [github.com/harbor302](https://github.com/harbor302)  
- **Summary:** Machine learning engineer specializing in computer vision, graph anti-fraud, and multimodal generation. Incoming MSCS student at NYU Courant (’27) seeking a Summer 2026 MLE internship.

## Education

- **New York University – Courant Institute of Mathematical Sciences** · Master of Science in Computer Science · Sept 2025 – May 2027 (expected) · New York, NY  
- **China University of Petroleum (Beijing)** · Bachelor of Engineering in Artificial Intelligence · Sept 2021 – Jun 2025 · Beijing, China

## Work Experience

### Machine Learning Engineer Intern · Baidu, Inc. · Beijing, China · Nov 2023 – Nov 2024
- Developed FFmpeg + MTCNN preprocessing, CosFace embeddings, and ANN clustering to detect duplicate faces, enabling automation of 1.6k+ takedowns/day.
- Built semi-supervised anti-fraud models (SMOTE, LOF, stacking, TrAdaBoost, self-training) combining ASR/OCR signals, fingerprint similarity, and profile features; recall improved from 74% → 92%, precision from 75% → 96%.
- Deployed GraphSAGE + Louvain community detection and XGBoost/GBDT/RF risk scoring on Kubernetes with MLflow tracking and A/B evaluation.

### Machine Learning Engineer Intern · China National Petroleum Corporation · Beijing, China · Jun 2024 – Jul 2024
- Automated OpenCV preprocessing and SAM-based mask generation for 12k images; fine-tuned SAM + Faster R-CNN raising mAP@[0.5:0.95] to 42.4.
- Dockerized and distributed training across a Hadoop cluster to reduce epoch time by 35% and deliver a hardened anomaly detection workflow.

### Machine Learning Engineer Intern · Ragine Technology (Incubated by Xidian University) · Xi’an, China · Jul 2023 – Sep 2023
- Expanded datasets from 1k → 15k with mosaic augmentation and synthetic occlusion; re-clustered YOLOv8 anchors and tuned NMS/IoU for IR targets.
- Combined U-Net segmentation with OSTrack tracking and deployed the full stack to embedded edge devices for drone imagery, maintaining real-time throughput.

## Research & Leadership

### CV Group Lead · CUPB Robotics Team (RoboMaster & Robocon Champion) · Beijing, China · Sep 2022 – Jun 2025
- Led vision subsystem integrating RGB-D sensors, LiDAR calibration, pose estimation, and coordinate transforms; computed depth-based yaw via YOLOv5 + Kalman filtering in ROS2 with serial I/O.
- Built TD-learning policies with ε-greedy exploration, early pruning, and hash-based deduplication; automated deployment via OpenVINO on Ubuntu/Linux NUCs with Shell-based rollback.
- Created Webots simulation environments for data collection, testing, and reinforcement learning iterations.

### Research Assistant · Multimodal Sentiment Classification & Generation · CUPB · Jan 2025 – May 2025
- Fused ResNet-50 visual embeddings and RoBERTa text features through Transformer cross-attention, surpassing the MMBT baseline by 8% accuracy.
- Fine-tuned Stable Diffusion with LoRA (rank 8) to enable emotion-conditioned image generation for reporting and internal demos.

### Research Assistant · Xidian University · Jul 2023 – Sep 2023
- Scaled industrial datasets, re-calibrated YOLOv8 anchors, and combined U-Net segmentation with OSTrack tracking for embedded Linux deployment.

## Skills

- **Languages:** Python (Pandas, NumPy, scikit-learn, Matplotlib), C/C++, Bash/Shell, SQL, Racket, HTML/CSS, CUDA  
- **Developer Tools:** GitHub, GitLab, MLflow, Docker, Kubernetes, Flask, AWS, Azure, GCP, Hadoop, Spark, Xmind  
- **Frameworks/Models:** PyTorch, TensorFlow, Keras, Hugging Face, OpenCV, YOLOv5/v8, U-Net, SAM, GraphSAGE, Transformers, Stable Diffusion, LLaMA, GPT  

## Publication

- **CaSTGCN: Deep Learning Method for Information Cascade Prediction** · IEEE CAC, Feb 2025  
  Proposed TCN + ST-GCN with dual-branch 1×1 attention, Optuna (400 trials) for hyperparameter search, and training refinements (MultiStepLR, normalization, dropout, residuals, warmup, early stopping) that cut MSLE by 15% versus baselines.
