import os
import requests
import torch
from tqdm import tqdm

MODEL_URL = "https://drive.google.com/uc?id=YOUR_GDRIVE_FILE_ID"  # You'll need to update this
MODEL_PATH = 'models/plant_disease_model.pth'
CLASS_INDICES_PATH = 'models/class_indices.json'

def download_model():
    """Download the model file if it doesn't exist."""
    os.makedirs('models', exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print("Downloading model file...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(MODEL_PATH, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            return False
    return True

def load_model(device, num_classes):
    """Load the model and return it."""
    try:
        if not os.path.exists(MODEL_PATH):
            if not download_model():
                return None
                
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        from train_model import create_model  # Import here to avoid circular import
        model = create_model(num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None 