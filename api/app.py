# Basic Flask app to serve as the API for Alzheimer's Disease Detection
from model.alzheimers_model import AlzheimerNet
import torch
from torchvision import transforms
from PIL import Image
import io
from flask import Flask, request, jsonify
import os

# Assume your model is in a file called model.py

app = Flask(__name__)

# Load the model
model = AlzheimerNet(num_classes=4)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    script_dir, '..', 'saved_models', 'alzheimer_model.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()


# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route('/')
def home():
    return "Alzheimer's Disease Detection API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read the image file in PIL format
        image = Image.open(io.BytesIO(file.read()))

        # Convert image to RGB if it's not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply the defined transformations
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Forward pass, get the model output
        with torch.no_grad():
            output = model(image_tensor)

        # Get the predicted class label
        _, predicted = torch.max(output.data, 1)

        # Assuming the classes are labeled as per your dataset classes
        classes = ['nondemented', 'very mild',
                   'mild demented', 'moderate demented']
        prediction_label = classes[predicted.item()]

        return jsonify({'result': prediction_label}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)
