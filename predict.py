import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

from model import build_hybrid
from config import get_config, get_weights_file_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = get_config()

model = build_hybrid(config)
model.to(device)
model_filename = get_weights_file_path(config, config['preload'])
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class_names = [
    'Adenocarcinoma',
    'Benign',
    'Squamous Cell Carcinoma'
]


test_folder = "testing"

image_paths = []
true_labels = []

for class_name in class_names:
    class_path = os.path.join(test_folder, class_name)
    for img in os.listdir(class_path):
        image_paths.append(os.path.join(class_path, img))
        true_labels.append(class_name)


plt.figure(figsize=(12, 12))

for i in range(min(9, len(image_paths))):

    img_path = image_paths[i]
    true_label = true_labels[i]

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_label = class_names[predicted.item()]
    conf_score = confidence.item()

    color = "green" if predicted_label == true_label else "red"

    plt.subplot(3, 3, i+1)
    plt.imshow(image)
    plt.title(
        f"True: {true_label}\nPred: {predicted_label}\nConf: {conf_score:.2f}",
        color=color,
        fontsize=9
    )
    plt.axis("off")

plt.tight_layout()
plt.show()