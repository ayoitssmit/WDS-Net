# WDS-Net: Multi-Path Feature Disentanglement Framework

WDS-Net is a general-purpose, PyTorch-based Deep Learning framework designed for image classification tasks. It leverages a unique multi-path architecture that simultaneously extracts Spatial (CNN), Structural (LSTM), and Global (Statistical) features from input images to improve predictive performance. 

While originally tailored for Devanagari characters and Hindi-MNIST numerals, the framework is generalized to train on **any image dataset**.

## 🌟 Key Features
- **Parallel Feature Disentanglement**:
  - **Spatial Path**: Extracts scale-invariant features via Deep CNNs and Max Pooling.
  - **Structural Path**: Captures sequential dependencies (stroke orders/structural relationships) using an LSTM over the Convoluted feature slices.
  - **Global Path**: Statically computes Mean, Variance, and Flattened Intensity Histograms.
- **Unified Fusion Module**: Features are concatenated and processed through specialized FC (Fully Connected) dense blocks.
- **Robust Evaluation Engine**: Automatically computes Accuracy, Precision, Recall, F1-Scores, Multi-class ROC-AUC curves, and generates Confusion Matrix heatmaps.

---

## 🚀 Getting Started

### 1. Requirements

Ensure you have Python 3.8+ installed. 

**For GPU Acceleration (Strongly Recommended)**:
You must install the version of PyTorch compiled for CUDA. If you currently have the CPU version installed, uninstall it first:
```bash
pip uninstall torch torchvision torchaudio
```
Then install the GPU-enabled version (defaulting to CUDA 11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Install Remaining Dependencies**:
```bash
pip install numpy opencv-python scikit-learn matplotlib seaborn tqdm
```

### 2. Preparing Your Dataset

The `UniversalDataset` loader is designed to automatically infer classes from your directory structure. You can supply multiple root directories (e.g., combining characters and digits).

**Expected Directory Structure**:
```text
your_dataset_path/
├── Train/
│   ├── class_A/
│   │   ├── img1.png
│   │   └── img2.png
│   ├── class_B/
│   └── class_C/
└── Test/
    ├── class_A/
    ├── class_B/
    └── class_C/
```
*Note: The script handles image resizing (defaults to 28x28), grayscale conversion, denoising (Gaussian Blur), and normalization automatically.*

### 3. Training the Model

Simply point the entry script to your training and testing directories:

```bash
python main.py --train_dirs "path/to/Dataset1/Train" "path/to/Dataset2/Train" --test_dirs "path/to/Dataset1/Test" "path/to/Dataset2/Test"
```

**Optional Hyperparameters & Checkpointing**:
- `--epochs`: Number of training iterations (default: `10`).
- `--batch_size`: Mini-batch size (default: `32`).
- `--lr`: Adam optimizer learning rate (default: `0.001`).
- `--device`: Target processing unit (`cuda` or `cpu`). The script prioritizes GPU if correctly installed.
- `--save_path`: Where to save the final `pth` state dictionary.
- `--checkpoint_path`: Where to save/load intermediate epoch training checkpoints.
- `--resume`: Add this flag to automatically resume training from the latest checkpoint if training was interrupted.

Example Resuming Command:
```bash
python main.py --train_dirs "path/Train" --test_dirs "path/Test" --resume
```

---

## 📁 Project Structure

- `main.py`: The CLI entry point orchestrating dataset loading, training, and testing.
- `model.py`: Contains the `WDSNet` PyTorch architecture (CNN, LSTM, Fusion).
- `dataset.py`: Houses the `UniversalDataset` class for generalized directory parsing and preprocessing.
- `train.py`: The core training loop applying the Categorical Cross-Entropy criterion and Adam optimizer.
- `evaluate.py`: The testing execution loop capturing logic for multi-class metrics (Scikit-Learn).
- `utils.py`: Helper formulas for statistical Global Feature extraction and matplotlib plotting.

---

## 📊 Outputs & Visualizations

After training completes, the framework performs inference on the test set and drops two visualization files in your root directory:
1. `training_curves.png`: Loss decay and validation accuracy growth over epochs.
2. `confusion_matrix.png`: A Seaborn heatmap detailing true vs. predicted class overlap to highlight structural similarities.
3. `roc_curves.png`: AUC analysis mapping FPR vs TPR (if enabled in `evaluate.py`).
