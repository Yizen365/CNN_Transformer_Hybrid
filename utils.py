import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np
import seaborn as sns
import os

def plot_confusion_matrix(cm, class_names, epoch, save_dir="outputs/confusion_matrices"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - Epoch {epoch}")

    save_path = os.path.join(save_dir, f"confusion_matrix_epoch_{epoch}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Confusion matrix saved at {save_path}")


def plot_roc_curves(all_labels, all_probs, class_names, epoch, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Convert labels to one-hot format
    n_classes = len(class_names)
    binary_labels = label_binarize(all_labels, classes=range(n_classes))

    plt.figure(figsize=(7, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - Epoch {epoch}")
    plt.legend(loc="lower right")

    save_path = os.path.join(save_dir, f"roc_epoch_{epoch}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"ROC curves saved at {save_path}")