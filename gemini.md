# WDS-Net: Multi-Path Feature Disentanglement Framework

## Architectural Overview
WDS-Net is designed to process 28x28 normalized grayscale images of Hindi characters and numerals through three parallel learning paths. The spatially invariant, structural, and global features extracted from these parallel streams are fused together to classify the inputs robustly.

## Mathematical Formulas

1. **Spatial Path (Deep CNN):**
   f^(l) = sigma(W^(l) * f^(l-1) + b^(l))
   Where * denotes convolution and sigma represents the ReLU activation function.

2. **Structural Path (LSTM):**
   h_t = LSTM(s_t, h_{t-1})
   Where s_t is the flattened spatial sequence derived from CNN feature maps at time step t.

3. **Global Path (Statistical):**
   F_global = [mean(x_i'), variance(x_i'), hist(x_i')]
   This represents the mean, variance, and flattened intensity histogram directly computed from the preprocessed input.

4. **Fusion via Concatenation:**
   F_unified = F_spatial + F_structural + F_global
   The unified feature map is processed by fully connected dense layers.

## Dataset Expectations
- **Inputs:** Combined Devanagari numerals dataset and a broader Hindi character dataset.
- **Grayscale Normalization:** x_i' = (x_i - mu) / sigma
- **Resizing & Denoising:** Resized to H = W = 28. Noise suppression (e.g., Gaussian Blur) applied to reduce stroke irregularities.

## Evaluation Strategy
1. **Training Parameters:** Adam Optimizer with adaptive learning rates. Categorical Cross-Entropy Loss to measure the discrepancy.
2. **Metrics:** Accuracy, Precision, Recall, and F1-Score metrics to evaluate performance.
3. **Visualizations and Outputs:**
   - Confusion Matrix mapping actual vs predicted classes to examine structural similarities.
   - ROC-AUC Curve computing True Positive vs False Positive rates for class separation capabilities.
   - Accuracy and Loss Training curves.
