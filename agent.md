# Agent Internal State Tracker

## Current Tasks
- [x] Extract rules and objective from the prompt.
- [x] Create project living documentation (gemini.md).
- [x] Define agent state tracker (agent.md).
- [x] Implement PyTorch WDS-Net modular script with spatial, structural, and global paths.

## Implemented Features
- **Modular Project Structure:** Codebase divided into `main.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, and `utils.py`.
- **WDS-Net Model Architecture:** Defines Spatial (CNN), Structural (LSTM) and Global paths. Fuses features via concatenation and fully connected layers.
- **Data Preprocessing:** Handled by Dataset subclass (Resizing to 28x28, Denoising, Normalization, Histogram matching).
- **Evaluation Loop:** Features functions to evaluate Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC curves.
- **Training Loop:** Adam optimizer and categorical cross-entropy criterion initialized.

## Pending Bugs
- None documented at this stage. 

## Next Steps
- [x] Acquire and place authentic Devanagari numeral and character image data into directory structures. (Done! Data structured and combined dataset loader implemented).
- [ ] Run `python main.py` to initiate the first training and evaluation loop.
- [ ] Tune the CNN depth, LSTM sequence dimensions, or learning rate based on initial benchmarking loss metrics.
- [ ] Extract and analyze the generated Confusion Matrix imagery and ROC-AUC plots post-training.
