from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from model_manager import load_model, CLASS_INDICES_PATH

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Get port from environment variable for deployment
PORT = int(os.environ.get('PORT', 5000))

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load class indices and model
try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    num_classes = len(class_to_idx)
    model = load_model(DEVICE, num_classes)
    if model:
        print("Model loaded successfully!")
    else:
        print("Failed to load model")
except Exception as e:
    print(f"Error loading model or class indices: {str(e)}")
    model = None
    idx_to_class = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    # Define the same transforms as used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess the image
            img_tensor = preprocess_image(filepath)
            img_tensor = img_tensor.to(DEVICE)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()
            
            # Get the class name
            predicted_class = idx_to_class.get(predicted_idx, 'Unknown')
            
            # Get disease information
            disease_info = get_disease_info(predicted_class)
            
            return jsonify({
                'success': True,
                'prediction': predicted_class,
                'confidence': confidence,
                'disease_info': disease_info
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Invalid file type'}), 400

def get_disease_info(disease_class):
    # Dictionary containing information about each disease
    disease_info = {
        'Apple___Apple_scab': {
            'description': 'Apple scab is a common disease of apple trees caused by the fungus Venturia inaequalis.',
            'symptoms': ['Dark olive-green spots on leaves', 'Dark, scabby lesions on fruits', 'Deformed fruits'],
            'treatment': ['Remove infected leaves', 'Apply fungicide in early spring', 'Maintain good air circulation']
        },
        'Apple___Black_rot': {
            'description': 'Black rot is a fungal disease that affects apple trees, caused by Botryosphaeria obtusa.',
            'symptoms': ['Purple spots on leaves', 'Rotting fruit with dark centers', 'Cankers on branches'],
            'treatment': ['Prune infected branches', 'Remove mummified fruits', 'Apply fungicide during growing season']
        },
        'Apple___Cedar_apple_rust': {
            'description': 'Cedar apple rust is caused by the fungus Gymnosporangium juniperi-virginianae.',
            'symptoms': ['Bright orange-yellow spots on leaves', 'Deformed fruits', 'Spots with black dots in the center'],
            'treatment': ['Remove nearby cedar trees', 'Apply fungicide preventatively', 'Plant resistant varieties']
        },
        'Apple___healthy': {
            'description': 'The plant appears to be healthy.',
            'symptoms': ['No visible symptoms of disease'],
            'treatment': ['Continue regular maintenance', 'Monitor for early signs of disease', 'Maintain good growing conditions']
        },
        # Add more disease information as needed
    }
    
    return disease_info.get(disease_class, {
        'description': 'Information not available',
        'symptoms': [],
        'treatment': []
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT) 