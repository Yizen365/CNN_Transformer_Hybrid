from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class LungCancerDataset(Dataset):

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # class names
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])

        # class -> index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # collect image paths
        self.samples = []

        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    label = self.class_to_idx[class_name]
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label