import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

from torchvision import transforms

from model import build_vit
from dataset import LungCancerDataset
from config import get_config, get_weights_file_path

from tqdm import tqdm


def run_validation(model, val_dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_dataloader:
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model.encode(images)
            projs = model.project(outputs)

            _, preds = torch.max(projs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    val_acc = 100 * correct/total
    print(f"Validation Accuracy: {val_acc:.2f}%")


def get_ds(config):

    dataset = LungCancerDataset(config['data_dir'])

    n = len(dataset)

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * n)
    val_ds_size = n - train_ds_size

    g = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(range(n), [train_ds_size, val_ds_size], generator=g)

    # Augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)), # Crop with slight scale/aspect variation for robustness
        transforms.RandomHorizontalFlip(p=0.5), # 50% chance; safe for isotropic medical images
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05), # Mild color shifts to handle staining variability
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet norms; replace with your dataset's if computed
    ])

    # No Augmentation
    val_transform = transforms.Compose([
        transforms.Resize(256), # Slight oversize for center crop
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transforms using Subset
    train_dataset = Subset(dataset, train_indices.indices)
    val_dataset = Subset(dataset, val_indices.indices)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    return train_dataloader, val_dataloader


def get_model(config):
    model = build_vit(config['image_size'], config['in_channels'], config['patch_size'], config['heads'], config['mlp_dim'], config['embedding_dim'], config['class_size'], config['layers'])
    return model


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader = get_ds(config)
    model = get_model(config).to(device)
    #Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        for images, labels in batch_iterator:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model.encode(images)
            projs = model.project(outputs)

            loss = loss_fn(projs, labels)
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        
        run_validation(model, val_dataloader, device)
        
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__=='__main__':
    config = get_config()
    train_model(config)