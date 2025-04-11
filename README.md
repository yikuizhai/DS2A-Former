# DS2A-Former
This is the official repository for “DS2A-Former: Battery Surface Defect Segmentation via Dual Stream Spatial Attention Transformer Network”. The repo is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

# Abstract
With the advancement of industrial automation, surface defect inspection has become crucial for battery manufacturing quality assurance. However, existing public datasets lack sufficient resources for battery surface defects, hindering the process of battery defect inspection. To address this, we developed the Multi-class Battery Surface Defect dataset (M-BSD), which contains 2,491 images and 6,639 samples spanning three defect types. Meanwhile, deep learning-based methods have made progress in industrial quality inspection but face challenges in handling complex battery defect details, defect category similarities, and small target defects. These issues hinder accurate, quantitative analysis of defects. To overcome above challenges, we propose the DS$^2$A-Former, a novel Dual Stream Spatial Attention Transformer Network that integrates semantic and spatial information to enhance segmentation performance under complex conditions. The Dual Scale Spatial Cross Attention (DSSCA) module facilitates information exchange across scales and suppresses background noise, while the Dual Query Gated Spatial Attention (DQGSA) module uses a spatial gating mechanism to prioritize knowledge between two queries, improving representation of defect information. Extensive experiments on the M-BSD dataset validate the effectiveness of our approach. We also experimented it on Crack-500 dataset, demonstrating its broad applicability in defect segmentation across domains.

# M-BSD Dataset
M-BSD dataset is available. The following is the method to apply for a dataset：

1. Use the school's email address (edu, stu, etc.) and send an email to: yikuizhai@163.com
2. Sign the relevant agreement to ensure that it will only be used for scientific research and not for commercial purposes.A scanned version of the agreement that requires a handwritten signature. Both Chinese and English signatures are acceptable.
3. Authorization will be obtained in 1-3 days. (Notice: If you use this dataset as the benchmark dataset for your paper, please cite the paper)
