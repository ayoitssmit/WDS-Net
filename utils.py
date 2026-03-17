import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def extract_global_features(img_norm):
    """
    Extracts global statistical descriptors (Mean, Variance, Intensity Histogram)
    from a normalized image.
    """
    mu = np.mean(img_norm)
    sigma = np.std(img_norm) + 1e-8
    
    # Range typically between -3 and 3 for normalized images
    hist, _ = np.histogram(img_norm, bins=16, range=(-3, 3), density=True)
    global_features = np.array([mu, sigma**2] + hist.tolist(), dtype=np.float32)
    return global_features

def plot_training_curves(train_losses, val_accuracies, save_path="training_curves.png"):
    """
    Plots training loss and validation accuracy curves.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    
    # Handle case where validation wasn't done
    if val_accuracies:
        ax2.set_ylabel('Validation Accuracy', color=color)
        ax2.plot(epochs, val_accuracies, color=color, marker='s', label='Val Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Training Loss and Validation Accuracy")
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, num_classes, save_path="confusion_matrix.png"):
    """
    Plots the confusion matrix.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(fpr_dict, tpr_dict, roc_auc_dict, num_classes, save_path="roc_curves.png"):
    """
    Plots ROC curves for multi-class classification.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot average or individual curves (simplified for clarity if many classes)
    for i in range(min(num_classes, 10)): # Plot up to 10 classes to avoid clutter
        if i in fpr_dict and i in tpr_dict:
            plt.plot(fpr_dict[i], tpr_dict[i], lw=2, label=f'Class {i} (AUC = {roc_auc_dict[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def save_checkpoint(model, optimizer, epoch, path="wds_net_checkpoint.pth"):
    """Saves the model state, optimizer state, and current epoch."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} at epoch {epoch}")

def load_checkpoint(model, optimizer, path="wds_net_checkpoint.pth", device='cpu'):
    """Loads model and optimizer states to resume training."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {path}. Resuming from epoch {epoch}")
    return model, optimizer, epoch

def save_model(model, path="wds_net_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="wds_net_model.pth", device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model
