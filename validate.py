import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score

from utils import plot_roc_curves


def run_validation(model, val_dataloader, device, loss_fn, epoch):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_dataloader:
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            loss = loss_fn(outputs, labels)  
            total_loss += loss.item()         
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total = len(all_labels)

    val_loss = total_loss / len(val_dataloader)

    val_acc = 100 * correct/total
    bal_acc = balanced_accuracy_score(all_labels, all_preds) * 100
    report = classification_report(all_labels, all_preds, target_names=['Adenocarcinoma', 'Benign', 'Squamous Cell'])
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Validation Accuracy (Patient Level): {val_acc:6.3f}%")
    print(f"Balanced Accuracy: {bal_acc:6.3f}%")
    print(report)

    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    class_names = ['Adenocarcinoma', 'Benign', 'Squamous cell']
    plot_roc_curves(all_labels, all_probs, class_names, epoch)
    
    return macro_f1, bal_acc, conf_matrix, val_loss