import os
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'Structureclassification.pkl'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained model
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
            return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

# Ensure the model is loaded successfully
if model is None:
    # Handle the case where the model is not loaded
    @app.route('/')
    def index():
        return render_template('index.html', error_message="Model loading failed. Please check the model path and try again.")

    @app.route('/predict', methods=['POST'])
    def predict():
        return jsonify({'error': 'Model loading failed. Please check the model path and try again.'}), 500

    app.run(debug=True)
    exit()

# ... rest of the code ...

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'Structureclassification.pkl'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads folder if it doesn't exist

# Load pre-trained model (handle potential errors)
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

# Damage classification labels
DAMAGE_LABELS = [
    'No Damage',
    'Minor Damage',
    'Moderate Damage',
    'Heavy Damage'
]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Adjust size to match model's expected input
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess image (handle potential errors)
            processed_image = preprocess_image(filepath)
            if processed_image is None:
                return jsonify({'error': 'Error preprocessing image'}), 500

            # Make prediction
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = float(np.max(predictions))

            # Clean up uploaded file
            os.remove(filepath)

            return render_template("index.html", prediction_text=f"Damage Classification: {DAMAGE_LABELS[predicted_class]} with confidence {confidence * 100:.2f}%")

        except Exception as e:
            print(f"Error making prediction: {e}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)