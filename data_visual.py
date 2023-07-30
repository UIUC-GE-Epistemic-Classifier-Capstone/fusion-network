import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import random
from dataset import get_dataset
from torch.utils.data import DataLoader,random_split
from network import FusionNet
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

train_ratio = 0.8
batch_size = 5

dataset = get_dataset()
print("dataset:", dataset)
train_size = int(len(dataset) * train_ratio)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloaders = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloaders = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model = FusionNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.000001, momentum=0.9)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Decrease lr by a factor of 0.1 every 10 epochs

def read_feature_csv(file_path):
    features = []
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            feature = row
            features.append([float(i) for i in feature])
    return torch.from_numpy(np.array(features[1:]))
    
def display_images_and_predictions(features, rgb_dir='./data/visible', ir_dir='./data/thermal'):
    softmax = nn.Softmax(dim=0)
    for i, feature in enumerate(features):
        # Apply softmax to the RGB and IR car and person features
        rgb_features = feature[:2]
        ir_features = feature[3:5]

        rgb_softmax_values = feature[6]
        ir_softmax_values = feature[7]

        rgb_class = "car" if rgb_softmax_values == 0  else "person"
        ir_class = "car" if ir_softmax_values == 0 else "person"

        # Load corresponding images
        rgb_image_path = os.path.join(rgb_dir, f'{i+1:05d}.jpg')  # Adjust image filename format if needed
        ir_image_path = os.path.join(ir_dir, f'{i+1:05d}.jpg')    # Adjust image filename format if needed

        rgb_image = Image.open(rgb_image_path)
        ir_image = Image.open(ir_image_path)

        rgb_image = np.array(rgb_image)
        ir_image = np.array(ir_image)

        # Create subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Show RGB image
        axs[0].imshow(rgb_image)
        axs[0].set_title(f"RGB image\nReliability: {feature[2]}\nClass: {rgb_class}")

        # Show IR image
        axs[1].imshow(ir_image)
        axs[1].set_title(f"IR image\nReliability: {feature[5]}\nClass: {ir_class}")

        # Show fusion prediction
        fusion_prediction = model(feature.unsqueeze(0)).argmax(dim=1).item()
        fusion_class = "car" if fusion_prediction == 0 else "person"

        axs[2].text(0.5, 0.5, f'Fusion Prediction: {fusion_class}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=15,
                    bbox=dict(facecolor='red', alpha=0.5))
        axs[2].axis('off')
        # plt.show()
        
        # Create directory if not exists
        os.makedirs('./data_visual', exist_ok=True)
        # Save the figure
        fig.savefig(f'./data_visual/figure_{i+1:05d}.png')


# Load the trained model
model.load_state_dict(torch.load("saved_model.pt"))
model.eval()

features = read_feature_csv('./feature.csv')
display_images_and_predictions(features)
