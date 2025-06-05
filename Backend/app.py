from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import base64
import json
from utils import transform_image
from torchvision.models import resnet18

app = Flask(__name__)
CORS(app)

# Load class mapping from training
with open("class_to_idx.json", "r") as f:
    idx_to_class = {v: k for k, v in json.load(f).items()}
class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

# Define model
class CurrencyClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet18()
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Load model
model = CurrencyClassifier(num_classes=len(class_labels))
state_dict = torch.load("currency_model.pth", map_location="cpu")
state_dict = {"resnet." + k: v for k, v in state_dict.items()}  # Add prefix
model.load_state_dict(state_dict)
model.eval()

# Prediction endpoint
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
        outputs = model(tensor)

        probs = torch.nn.functional.softmax(outputs, dim=1)
        print("Probabilities:", probs.tolist())  # Optional: see prediction confidence

        _, predicted = torch.max(outputs.data, 1)
        label = class_labels[predicted.item()]

        return jsonify({"result": label})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
