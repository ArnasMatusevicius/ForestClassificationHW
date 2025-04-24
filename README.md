# SimpleUNet - Image Segmentation

This repository provides a straightforward PyTorch implementation of a simplified UNet architecture for image segmentation tasks.

## Overview

SimpleUNet is a minimalistic yet powerful convolutional neural network designed for semantic segmentation. It efficiently encodes image features and decodes them into pixel-level class predictions.

## Installation

Clone this repository and install dependencies using:

```bash
git clone <repo-url>
cd simple-unet
pip install torch torchvision numpy matplotlib opencv-python
```

## Usage

### Model Initialization

Initialize the SimpleUNet model:

```python
import torch
from simple_unet import SimpleUNet

model = SimpleUNet(num_classes=3)  # Adjust num_classes as per your use case
```

### Running Inference

Perform inference on an input image:

```python
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# Load and preprocess image
img = cv2.imread('path/to/image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])

input_tensor = transform(img_rgb).unsqueeze(0)

# Model inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)

# Generate prediction mask
probabilities = torch.softmax(output, dim=1)
predicted_mask = probabilities.argmax(dim=1).squeeze().cpu().numpy()

# Visualization
plt.imshow(predicted_mask, cmap='jet')
plt.title("Segmentation Output")
plt.axis('off')
plt.show()
```

## Training

Example training loop:

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    optimizer.zero_grad()

    output = model(input_tensor)
    loss = criterion(output, ground_truth_tensor)  # Ground truth shape: (B, H, W)

    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
```

## Saving and Loading Models

Save your trained model:

```python
torch.save(model.state_dict(), 'simple_unet.pth')
```

Load a saved model:

```python
model.load_state_dict(torch.load('simple_unet.pth'))
model.eval()
```


