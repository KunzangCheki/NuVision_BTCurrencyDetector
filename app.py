from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import base64
from torchvision import transforms
from PIL import Image
import io
from torchvision.models import resnet18

app = Flask(__name__)
CORS(app)

# Define class labels explicitly in the order you used in Colab
class_labels = ['Nu.1', 'Nu.10', 'Nu.100', 'Nu.1000', 'Nu.20', 'Nu.5', 'Nu.50', 'Nu.500']


image_size = 256  # or the same size used during training

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)


class CurrencyClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Load model weights
model = CurrencyClassifier(num_classes=len(class_labels))
state_dict = torch.load("currency_model.pth", map_location="cpu")

# Fix prefix mismatch if needed
if not any(k.startswith("resnet.") for k in state_dict.keys()):
    state_dict = {f"resnet.{k}": v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or 'data' not in data or not data['data']:
            return jsonify({"error": "No valid image data provided"}), 400

        image_b64 = data['data'][0]
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        image_bytes = base64.b64decode(image_b64)

        tensor = transform_image(image_bytes)

        with torch.no_grad():
                outputs = model(tensor)
                print("Raw outputs:", outputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(probs, dim=1).item()
                label = class_labels[predicted_idx]


        # Debug logs
        print("Class Labels:", class_labels)
        print("Probabilities:", [round(p, 5) for p in probs[0].tolist()])
        print("Predicted Index:", predicted_idx)
        print("Predicted Label:", label)

        return jsonify({"result": label})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
