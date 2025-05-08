from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import torch
import numpy as np
from torchvision import models, transforms as T
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

app = Flask(_name_)
CORS(app)  # To allow cross-origin requests (needed for frontend and backend to communicate)

# Load the trained Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

# Pretrained CNN (ResNet18) for Grad-CAM
model_cnn = models.resnet18(pretrained=True).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cnn = model_cnn.to(device)

# Define image transformations for Grad-CAM
transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']
        
        # Predict with the Random Forest model
        prediction = rf_model.predict([features])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/gradcam', methods=['POST'])
def gradcam():
    try:
        # Get uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image part'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Open and transform the image
        image = Image.open(file).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Set Grad-CAM target layer and initialize Grad-CAM
        target_layer = model_cnn.layer4[-1]
        cam = GradCAM(model=model_cnn, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(0)]  # Dummy target class for visualization

        # Generate Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        rgb_image = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

        # Save the Grad-CAM image
        output_dir = "static"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        grad_cam_image_path = os.path.join(output_dir, "grad_cam_image.png")
        Image.fromarray(visualization).save(grad_cam_image_path)

        # Return the Grad-CAM image path
        return jsonify({'image_path': grad_cam_image_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if _name_ == '_main_':
    app.run(debug=True)
