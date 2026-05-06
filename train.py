import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

from model import build_hybrid
from config import get_config, get_weights_file_path
from get_dataset import get_ds
from validate import run_validation
from utils import plot_confusion_matrix

from tqdm import tqdm



# def set_backbone_grad(model, requires_grad: bool):
#     for param in model.parameters():
#         param.requires_grad = requires_grad


def get_model(config):
    model = build_hybrid(config)
    return model


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader = get_ds(config)
    model = get_model(config).to(device)

    #Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # Freeze the CNN
    # set_backbone_grad(model.cnn_backbone.cnn, requires_grad=False)

    optimizer = torch.optim.Adam([{"params": filter(lambda p: p.requires_grad, model.parameters()),
                                    "lr": config['learning_rate']}])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # ----- Class weights (give higher weight to rare classes) ------
    # counts = torch.tensor([config['benign_count'], config['malignant_count']], dtype=torch.float)
    # weights = 1.0 / torch.sqrt(counts)
    # weights = weights / weights.sum() * len(counts)
    # weights = weights.to(device)
    # ----------------------------------------------------------------

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_macro_f1 = 0.0

    for epoch in range(initial_epoch, config['num_epochs']):
        # if epoch == config['cnn_unfreeze_epoch']:
        #     set_backbone_grad(model.cnn_backbone.cnn, requires_grad=True)
        #     optimizer = torch.optim.Adam([
        #         {"params": model.cnn_backbone.cnn.parameters(), "lr":config['fine_tune_lr']},
        #         {"params": model.cnn_backbone.proj.parameters(), "lr": config['learning_rate']},
        #         {"params": [p for n, p in model.named_parameters() if not n.startswith("cnn_backbone")], "lr": config['learning_rate']}
        #     ])
        #     # Re-initialise scheduler for fine-tune phase
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'] - epoch, eta_min=1e-6)

        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in batch_iterator:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backpropagate the loss
            loss.backward()

            # Gradient clipping to prevent exploding gradients in hybrid model
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update the weights
            optimizer.step()

             # Training accuracy calculation (Slice level)
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                running_loss += loss.item() * images.size(0)

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.flush()

            global_step += 1
            
        scheduler.step()

        train_loss = running_loss / total
        train_acc = 100 * correct / total
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc (Slice level): {train_acc:.3f}%")
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/accuracy',train_acc, epoch)

        macro_f1, bal_acc, conf_matrix, val_loss = run_validation(model, val_dataloader, device, loss_fn, epoch)
        print(conf_matrix)

        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", bal_acc, epoch)
        writer.add_scalar("val/macro_f1", macro_f1, epoch)

        if macro_f1 is not None and macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1

            class_names = ['Adenocarcinoma', 'Benign', 'Squamous cell']
            plot_confusion_matrix(conf_matrix, class_names, epoch)

            model_filename = get_weights_file_path(config, "best")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'macro_f1': macro_f1,
            }, model_filename)
            print(f"----- Saved best model (macro_f1={macro_f1:.3f}) -----")
        else:
            model_filename = get_weights_file_path(config, "latest")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)
    
    print(f"\nTraining complete. Best macro F1 score: {best_macro_f1:.3f}")
    writer.close()


if __name__=='__main__':
    config = get_config()
    train_model(config)