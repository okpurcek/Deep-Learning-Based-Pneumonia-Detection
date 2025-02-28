# Deep Learning-Based Pneumonia Detection

## Overview
This project focuses on developing and evaluating deep learning models for detecting pneumonia in chest X-ray images. It implements state-of-the-art architectures such as ResNet-18, VGG16, EfficientNet-B0, and a custom CNN, incorporating advanced training strategies, model interpretability techniques, and real-world validation using an independent dataset.

## Features
- **Multiple CNN Architectures**: Implementation and evaluation of ResNet-18, VGG16, EfficientNet-B0, and a custom CNN.
- **Data Augmentation**: Extensive preprocessing and augmentation techniques to improve model generalization.
- **Grad-CAM Visualization**: Model interpretability using heatmaps to highlight critical regions in X-ray images.
- **Cross-Validation**: Performance evaluation with various metrics, including accuracy, precision, recall, and AUC-ROC.
- **Real-World Validation**: Testing on an independent dataset to assess clinical applicability.
- **Regulatory Compliance**: Ensures adherence to data privacy regulations, including Turkish KVKK law.

## Dataset
The dataset consists of chest X-ray images collected from multiple sources, including the ChestX-ray8 database. The dataset is split into:
- **Training Set**: 5862 images (1341 Normal, 4521 Pneumonia)
- **Validation Set**: 1880 images (360 Normal, 1520 Pneumonia)
- **Test Set**: 3313 images (624 Normal, 2689 Pneumonia)

### Preprocessing
- Image resizing to 224×224 pixels
- Intensity normalization
- Contrast enhancement using adaptive histogram equalization
- Noise injection and random erasing for robustness

## Model Architectures
### 1. ResNet-18
- Implements residual connections to improve deep learning training.
- Achieves **94% accuracy** on the test set and **92% in real-world validation**.

### 2. VGG16
- A deep architecture with a simple convolutional pipeline.
- Achieves **92% accuracy** with high generalization ability.

### 3. EfficientNet-B0
- Optimized for accuracy and efficiency using compound scaling.
- Achieves **93% accuracy** with efficient feature extraction.

### 4. Custom CNN
- A lightweight model designed for real-time applications.
- Achieves **89% accuracy**, suitable for resource-constrained environments.

## Training Strategy
- **Optimizer**: AdamW for efficient weight decay handling.
- **Learning Rate Scheduling**: OneCycleLR for dynamic adjustments.
- **Loss Function**: Cross-entropy loss for binary classification.
- **Batch Size**: 32 for balancing memory usage and efficiency.
- **Gradient Clipping**: Prevents exploding gradients in deep networks.
- **Early Stopping**: Stops training when validation loss plateaus.

## Results and Analysis
| Model          | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---------------|---------|----------|--------|---------|--------|
| **ResNet-18** | 94%     | 95%      | 93%    | 94%     | 0.97   |
| **VGG16**     | 92%     | 93%      | 91%    | 92%     | 0.95   |
| **EfficientNet-B0** | 93% | 94% | 92% | 93% | 0.96 |
| **Custom CNN** | 89%    | 90%      | 88%    | 89%     | 0.92   |

## Model Interpretability
### Grad-CAM Visualizations
- Highlights important regions in X-ray images that contribute to model predictions.
- Provides explainability for clinical decision-making.

## Real-World Validation
- Models were tested on an independent dataset.
- ResNet-18 maintained high performance with **92% accuracy**, proving robustness for clinical use.

## Deployment Considerations
- The models can be integrated into **computer-aided diagnosis (CAD) systems**.
- Potential applications include **automated triaging in hospitals** and **remote diagnostics in underserved areas**.
- Requires **further testing with diverse patient demographics** before clinical deployment.

## Future Improvements
- **Enhancing dataset diversity** to improve model generalization.
- **Integration with multimodal data** (e.g., clinical history, symptoms) for better predictions.
- **Real-time deployment optimization** for hospital use.
- **Addressing ethical considerations** in AI-driven healthcare.

## References
- Research papers and sources used in model development are cited in the accompanying report.

## Contributors
- **Umut Çalıkkasap** – Istanbul Technical University
- **Oğuz Kağan Pürçek** – Istanbul Technical University

## License
This project is for research purposes only. Ensure compliance with **data privacy laws** before using patient data.

---
For more details, refer to the [project report](DL_Final_REPORT.pdf).
