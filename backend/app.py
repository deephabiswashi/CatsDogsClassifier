#!/usr/bin/env python3
# Monkey-patch for werkzeug.urls to provide url_quote
try:
    from werkzeug.urls import url_quote
except ImportError:
    from werkzeug.utils import quote as url_quote
    import werkzeug.urls
    werkzeug.urls.url_quote = url_quote

import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Configuration flag:
# If predictions seem reversed, set SWAP_LABELS to True.
SWAP_LABELS = False

# Load saved models and utilities from the models directory
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
svm_model = joblib.load(os.path.join(MODELS_DIR, 'svm_model.pkl'))
rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
lr_model = joblib.load(os.path.join(MODELS_DIR, 'lr_model.pkl'))
kmeans_model = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

# Load label encoder if exists, else create one
label_encoder_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
if os.path.exists(label_encoder_path):
    le = joblib.load(label_encoder_path)
else:
    le = LabelEncoder()
    # Set the order according to your training data.
    le.fit(["cats", "dogs"])
    joblib.dump(le, label_encoder_path)

cnn_model = load_model(os.path.join(MODELS_DIR, 'cnn_model.h5'))

# Define separate image sizes for CNN and traditional ML models
# (These must match the sizes used during training.)
CNN_IMG_SIZE = 128   # For CNN branch
ML_IMG_SIZE = 64     # For traditional ML branch

def preprocess_image_ml(image_bytes, for_cnn=False):
    """
    Preprocess image:
      - For CNN: Resize to (CNN_IMG_SIZE, CNN_IMG_SIZE), normalize to [0,1], and add a batch dimension.
      - For traditional ML models: Convert to grayscale, resize to (ML_IMG_SIZE, ML_IMG_SIZE), flatten, 
        scale using the loaded scaler, and ensure the output is a contiguous float64 array.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if for_cnn:
        img = cv2.resize(img, (CNN_IMG_SIZE, CNN_IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (ML_IMG_SIZE, ML_IMG_SIZE))
        img = img.flatten().astype("float32")  # (ML_IMG_SIZE*ML_IMG_SIZE) features
        img = scaler.transform([img]).astype(np.float64)
        img = np.ascontiguousarray(img, dtype=np.float64)
    return img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'No image or model provided'}), 400
    file = request.files['image']
    model_choice = request.form['model']
    image_bytes = file.read()

    # For debugging: Print label classes (only in debug mode)
    app.logger.debug(f"Label encoder classes: {le.classes_}")
    
    if model_choice == 'cnn':
        processed = preprocess_image_ml(image_bytes, for_cnn=True)
        if processed is None:
            return jsonify({'error': 'Invalid image data'}), 400
        preds = cnn_model.predict(processed)
        label_index = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        result = le.inverse_transform([label_index])[0]
    else:
        processed = preprocess_image_ml(image_bytes, for_cnn=False)
        if processed is None:
            return jsonify({'error': 'Invalid image data'}), 400

        if model_choice == 'svm':
            pred = svm_model.predict(processed)
            proba = svm_model.predict_proba(processed).max()
        elif model_choice == 'rf':
            pred = rf_model.predict(processed)
            proba = rf_model.predict_proba(processed).max()
        elif model_choice == 'lr':
            pred = lr_model.predict(processed)
            proba = lr_model.predict_proba(processed).max()
        elif model_choice == 'kmeans':
            processed = np.array(processed, dtype=np.float64)
            processed = np.ascontiguousarray(processed, dtype=np.float64)
            pred = kmeans_model.predict(processed)
            # Here, use a simple mapping; if the mapping seems reversed, update it accordingly.
            mapping = {0: 'cats', 1: 'dogs'}
            result = mapping.get(pred[0], 'Unknown')
            proba = 1.0
            return jsonify({'result': result, 'confidence': proba})
        else:
            return jsonify({'error': 'Invalid model choice'}), 400
        
        result = le.inverse_transform(pred)[0]
        confidence = float(proba)
    
    # If SWAP_LABELS is True, swap the result
    if SWAP_LABELS:
        result = 'dogs' if result == 'cats' else 'cats'
    
    return jsonify({'result': result, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
