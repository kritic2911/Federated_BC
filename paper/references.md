# Academic References & Literature Bibliography
**Federated Learning on Cross-Silo Digital Mammography (BTP Deliverable)**

This document catalogs foundational literature, benchmark datasets, federated optimization algorithms, systems/cryptographic protocols, explainability sanity checks, and explicitly defines our **literature gap and honest contributions**.

---

## 1. Literature Overview & Research Hook Matrix

The following table summarizes the key papers directly informing our design choices, their core ideas, and the specific gaps that our work addresses:

| # | Paper | Core Idea | What It Leaves Open (Our Hook) |
|---|---|---|---|
| **1** | **Roth et al. 2020**, *Federated Learning for Breast Density Classification: A Real-World Implementation* ([arXiv:2009.01871](https://arxiv.org/abs/2009.01871)) | Real multi-hospital FL; the federated model is compared with local-only training. | Breast density classification, not lesion finding; no spatial localization or systems stress tests. |
| **2** | **Jiménez-Sánchez et al. 2021**, *Memory-aware curriculum FL for breast cancer classification* ([arXiv:2107.02504](https://arxiv.org/abs/2107.02504)) | Curriculum-style training to cope with heterogeneous clinical sites. | Classification-only evaluation; does not evaluate systems overhead or label-family disparity. |
| **3** | *Real-world case study on mammography FL*, **CSBJ 2025** ([PMC11984999](https://pmc.ncbi.nlm.nih.gov/articles/PMC11984999/)) | Preprocessing and harmonization effects on cross-site generalization. | Motivates our domain-shift analysis; lacks explainability validation and real systems telemetry. |
| **4** | *Evaluating FL approaches for mammography under breast-density heterogeneity* ([arXiv:2605.09137](https://arxiv.org/abs/2605.09137), 2026) | Compares aggregation strategies under clinical heterogeneity. | No label-family mismatch (BI-RADS vs biopsy), no network failure injection. |
| **5** | **McMahan et al. 2017**, *FedAvg: Communication-Efficient Learning of Deep Networks* ([arXiv:1602.05629](https://arxiv.org/abs/1602.05629)) | Average locally trained weights, weighted by client sample count. | Uniform weighting vs sample weighting is one of our explicit ablations across imbalanced silos. |
| **6** | **Li et al. 2020**, *FedProx: Federated Optimization in Heterogeneous Networks* ([arXiv:1812.06127](https://arxiv.org/abs/1812.06127)) | A proximal regularization term ($\frac{\mu}{2}\|w - w_t\|^2$) limits local client drift. | Small gains are hard to detect with few seeds; evaluated across $\mu \in \{0.01, 0.1\}$. |
| **7** | **Li et al. 2021**, *FedBN: Federated Learning on Non-IID Features via Local Batch Normalization* ([arXiv:2102.07623](https://arxiv.org/abs/2102.07623)) | Keep batch-norm statistics local to handle scanner feature shift. | Requires trainable BN layers; our baseline sweep investigates frozen vs trainable BN trade-offs. |
| **8** | **Bonawitz et al. 2017**, *Practical Secure Aggregation for Privacy-Preserving ML* ([ACM CCS '17](https://doi.org/10.1145/3133956.3133982)) | Pairwise additive masking so the server only observes the aggregate sum $\sum y_u$. | Dropout recovery is costly; we measure how that interacts with the minimum-cohort guard ($k$). |
| **9** | **Ghosh et al. 2024**, *Mammo-CLIP: Vision-Language Pretraining for Mammography* ([arXiv:2405.12255](https://arxiv.org/abs/2405.12255)) | Image–report contrastive pretraining on large-scale digital mammograms. | Preprocessing-sensitive; EfficientNet backbone; treated as optional foundation init in our plan. |
| **10** | **Selvaraju et al. 2017**, *Grad-CAM: Visual Explanations from Deep Networks* ([arXiv:1610.02391](https://arxiv.org/abs/1610.02391)) | Class-discriminative heatmaps from convolutional feature activations. | Coarse spatial maps at $512\times512$; must be validated quantitatively against clinical annotations. |
| **11** | **Adebayo et al. 2018**, *Sanity Checks for Saliency Maps* ([arXiv:1810.03292](https://arxiv.org/abs/1810.03292), NeurIPS 2018) | Randomizing model weights should qualitatively and quantitatively destroy saliency maps. | Our explainability validation includes weight-randomization and label-permutation tests. |

---

## 2. Gap and Contribution (Positioned Honestly)

> [!IMPORTANT]
> **Positioning:** This project is primarily an **empirical and systems contribution**, not a claim of a novel theoretical machine learning algorithm. We evaluate federated learning as an end-to-end engineering and clinical pipeline under real hospital constraints.

Our contribution rests on three concrete pillars:

1. **Rigorous Per-Site & Worst-Site Evaluation Across Mixed Label Families:**
   - Previous studies routinely pool datasets or assume homogeneous labeling. We explicitly model and test the boundary between **screening classification (`birads`: VinDr, INbreast)** and **diagnostic pathological classification (`biopsy`: CMMD, CDD)**.
   - We report macro one-vs-rest AUC, binary class-2-vs-rest AUC, and worst-site AUC across stratified, patient-grouped splits (preventing patient leakage across views).
2. **Explainable Breast-Level Reports Validated Against Clinical Annotations:**
   - Rather than outputting an isolated class score, the system delivers structured clinical reports: 3×3 anatomical quadrant heat distribution, calibrated confidence bands (via temperature scaling), and calcification candidate density.
   - Spatial heatmaps are validated against ground-truth expert contours (INbreast mass polygons, VinDr finding boxes) using hit-rate metrics and total-variation distance, verified with Adebayo weight-randomization sanity checks.
3. **Real-Deployment Systems Measurements & Failure Telemetry:**
   - We emulate multi-hospital deployments using Docker containers shaped with Linux `tc netem` (Metro: 5 ms delay, 1 ms jitter; District: 120 ms delay, 35 ms jitter, 2% loss) over mutual TLS.
   - We quantify the real-world overhead of **Bonawitz pairwise additive masking (Secure Aggregation)** versus plaintext FedAvg in wire bytes, round duration, and jitter.
   - We characterize the **privacy-vs-reliability frontier** under the **Minimum-Cohort Guard ($k$)**, recording how correlated non-random district hospital dropouts trigger automatic round aborts and prevent privacy leakage.

---

## 3. Detailed Citations

### Federated Optimization & Aggregation
- **FedAvg**: McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017, PMLR 54:1273-1282. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629).
- **FedProx**: Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M.,损伤, A., & Smith, V. (2020). *Federated Optimization in Heterogeneous Networks.* MLSys 2020, 2:429-450. [arXiv:1812.06127](https://arxiv.org/abs/1812.06127).
- **FedBN**: Li, X., Jiang, M., Zhang, X., Kamp, M., & Dou, Q. (2021). *FedBN: Federated Learning on Non-IID Features via Local Batch Normalization.* ICLR 2021. [arXiv:2102.07623](https://arxiv.org/abs/2102.07623).

### Cryptographic Privacy & Systems Networking
- **Secure Aggregation**: Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B., Patel, S., Ramage, D., Segal, A., & Seth, K. (2017). *Practical Secure Aggregation for Privacy-Preserving Machine Learning.* ACM CCS 2017, 1175–1191. [DOI: 10.1145/3133956.3133982](https://doi.org/10.1145/3133956.3133982).
- **Mutual TLS (mTLS)**: Cooper, D., et al. (2008). *Internet X.509 Public Key Infrastructure Certificate Profile.* RFC 5280.

### Digital Mammography Clinical Datasets
- **VinDr-Mammo**: Nguyen, H. T., et al. (2023). *VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography.* Scientific Data, 10(1), 277. [DOI: 10.1038/s41597-023-01963-1](https://doi.org/10.1038/s41597-023-01963-1).
- **CMMD**: Cui, C., et al. (2021). *The Chinese Mammography Database (CMMD).* The Cancer Imaging Archive (TCIA). [DOI: 10.7937/tcia.5r05-0036](https://doi.org/10.7937/tcia.5r05-0036).
- **INbreast**: Moreira, I. C., et al. (2012). *INbreast: Toward a Full-field Digital Mammography Database.* Academic Radiology, 19(2), 236–248. [DOI: 10.1016/j.acra.2011.09.014](https://doi.org/10.1016/j.acra.2011.09.014).
- **CDD-CESM**: Khaled, R., et al. (2022). *Categorized Digital Database for Low Energy and Subtracted Contrast-Enhanced Spectral Mammography images.* Scientific Data, 9(1), 583. [DOI: 10.1038/s41597-022-01615-1](https://doi.org/10.1038/s41597-022-01615-1).

### Mammography FL Studies
- **Roth et al.**: Roth, H. R., et al. (2020). *Federated Learning for Breast Density Classification: A Real-World Implementation.* [arXiv:2009.01871](https://arxiv.org/abs/2009.01871).
- **Jiménez-Sánchez et al.**: Jiménez-Sánchez, A., et al. (2021). *Memory-aware Curricula for Federated Learning: A Study on Cross-Silo Breast Mammography.* MICCAI 2021. [arXiv:2107.02504](https://arxiv.org/abs/2107.02504).
- **CSBJ 2025 Case Study**: *Real-world case study on mammography FL*, Computational and Structural Biotechnology Journal, 2025. [PMC11984999](https://pmc.ncbi.nlm.nih.gov/articles/PMC11984999/).
- **Density Heterogeneity Study (2026)**: *Evaluating FL approaches for mammography under breast-density heterogeneity.* [arXiv:2605.09137](https://arxiv.org/abs/2605.09137).

### Explainability, Calibration & Architectures
- **Grad-CAM**: Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV 2017. [DOI: 10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74).
- **Sanity Checks for Saliency**: Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). *Sanity Checks for Saliency Maps.* Advances in Neural Information Processing Systems (NeurIPS 2018). [arXiv:1810.03292](https://arxiv.org/abs/1810.03292).
- **Temperature Scaling**: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
- **ResNet**: He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. [DOI: 10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90).
- **Mammo-CLIP**: Ghosh, S., et al. (2024). *Mammo-CLIP: Vision-Language Foundation Model for Full-field Digital Mammography.* [arXiv:2405.12255](https://arxiv.org/abs/2405.12255).
