# Plant Disease Detector

A web application that uses deep learning to detect plant diseases from images. Built with Flask, PyTorch, and React.

## Features

- Upload plant images for disease detection
- Real-time disease prediction using deep learning
- Detailed information about detected diseases
- User-friendly interface
- Responsive design

## Prerequisites

- Python 3.8 or higher
- Node.js and npm (for frontend development)
- PyTorch
- Flask
- Other Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/plant-disease-detector.git
cd plant-disease-detector
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Download the pre-trained model:
```bash
# The model will be downloaded automatically when running the application
```

## Running the Application

1. Start the Flask backend server:
```bash
cd backend
python app.py
```

2. In a new terminal, start the frontend server:
```bash
cd frontend
python -m http.server 8000
```

3. Open your web browser and navigate to:
```
http://localhost:8000
```

## Project Structure

```
plant-disease-detector/
├── backend/
│   ├── app.py              # Flask backend server
│   ├── train_model.py      # Model training script
│   └── organize_dataset.py # Dataset organization script
├── frontend/
│   ├── index.html         # Main HTML file
│   └── app.js            # React frontend application
├── data/
│   └── organized/        # Organized dataset
├── requirements.txt      # Python dependencies
└── README.md           # Project documentation
```

## Usage

1. Click "Select Image" to choose a plant image
2. The selected image will be displayed
3. Click "Analyze Image" to get the disease prediction
4. View the results, including:
   - Predicted disease
   - Confidence score
   - Disease description
   - Symptoms
   - Treatment recommendations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset from Mendeley
- PyTorch and torchvision for deep learning
- Flask for backend
- React for frontend 