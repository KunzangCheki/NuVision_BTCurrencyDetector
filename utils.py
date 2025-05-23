from torchvision import transforms
from PIL import Image
import io
import torch

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),   # 256x256, not 224x224
        transforms.ToTensor(),            # No normalization
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)
