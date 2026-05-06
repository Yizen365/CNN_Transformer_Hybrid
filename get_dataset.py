import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import LungDataset


def get_ds(
    config,
    test_size=0.3,
    num_workers=4
):

    image_paths = []
    labels = []
    class_names = sorted(os.listdir(config['data_dir']))

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls_name in class_names:
        cls_folder = os.path.join(config['data_dir'], cls_name)
        for img_name in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_name)
            image_paths.append(img_path)
            labels.append(class_to_idx[cls_name])

    # Stratified split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=42
    )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = LungDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = LungDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader