---
title: "CaSTGCN: Deep Learning Method for Information Cascade Prediction"
collection: publications
category: conferences
permalink: /publication/2025-castgcn
excerpt: 'Combined TCN, spatio-temporal GCN, and dual-branch 1×1 attention with Optuna tuning to reduce MSLE by 15% on cascade forecasting.'
date: 2025-02-13
venue: 'IEEE Conference on Automation and Computing (CAC)'
paperurl: ''
citation: 'Harbor Liu, et al. (2025). "CaSTGCN: Deep Learning Method for Information Cascade Prediction." IEEE CAC.'
---

We propose **CaSTGCN**, a cascade forecasting model that fuses temporal convolutional networks with spatio-temporal graph convolutions and a dual-branch 1×1 attention module. Compared with strong baselines, CaSTGCN delivers a 15% MSLE reduction by:

- Running **Optuna** (400 trials) for hyperparameter search spanning receptive fields, attention sizes, and learning-rate schedules.  
- Employing training refinements—MultiStepLR, normalization, dropout, residual links, warmup, gradient clipping, and early stopping—for stable optimization.  
- Modeling both structural diffusion paths and temporal burstiness, enabling accurate long-horizon cascade prediction.

The implementation is built in **PyTorch** with reproducible experiment tracking. Please reach out if you’d like to collaborate or reproduce the study.
