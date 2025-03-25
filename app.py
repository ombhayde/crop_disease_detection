# # app.py
# from flask import Flask, render_template, request, redirect, url_for
# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Load model and class names
# model = load_model('models/saved_models/crop_disease_model.keras')
# with open('models/saved_models/class_names.txt', 'r') as f:
#     class_names = [line.strip() for line in f.readlines()]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return redirect(request.url)
    
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
    
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(filepath)
    
#     # Process image and predict
#     img = image.load_img(filepath, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
    
#     predictions = model.predict(img_array)[0]
#     predicted_class = class_names[np.argmax(predictions)]
#     confidence = float(predictions[np.argmax(predictions)])
    
#     return render_template('result.html', 
#                           filename=file.filename,
#                           predicted_class=predicted_class,
#                           confidence=confidence*100)

# if __name__ == '__main__':
#     app.run(debug=True)


# # app.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes
# app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Load model and class names
# model = load_model('models/saved_models/crop_disease_model.keras')
# with open('models/saved_models/class_names.txt', 'r') as f:
#     class_names = [line.strip() for line in f.readlines()]

# @app.route('/api/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(filepath)
    
#     # Process image and predict
#     img = image.load_img(filepath, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
    
#     predictions = model.predict(img_array)[0]
    
#     # Get top 3 predictions
#     top_indices = predictions.argsort()[-3:][::-1]
#     top_predictions = [
#         {"class": class_names[i], "confidence": float(predictions[i])} 
#         for i in top_indices
#     ]
    
#     result = {
#         "predicted_class": class_names[np.argmax(predictions)],
#         "confidence": float(predictions[np.argmax(predictions)]),
#         "top_predictions": top_predictions,
#         "image_url": f"/uploads/{file.filename}"
#     }
    
#     return jsonify(result)

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)


# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import time
import re


# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://crop-ai-frontend.onrender.com",  # Frontend URL
            "http://localhost:3000",  # Local development
            "*"  # Use cautiously, preferably specify exact origins
        ]
    }
})  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and class names
model = load_model('models/saved_models/crop_disease_model.keras')
with open('models/saved_models/class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Function to fetch remedies
def fetch_disease_remedy(disease_name):
    try:
        # Format the query to search for disease treatment
        query = f"{disease_name} plant disease treatment remedy"
        
        # Search for relevant pages (limit to agricultural/gardening domains for better results)
        search_results = list(search(query, num_results=5, lang="en"))
        
        # Filter for reliable agricultural domains if possible
        reliable_domains = ['.edu', '.gov', 'extension.', 'gardening.', 'agriculture.']
        filtered_results = [url for url in search_results if any(domain in url for domain in reliable_domains)]
        
        # Use filtered results if available, otherwise use original results
        urls_to_check = filtered_results if filtered_results else search_results[:3]
        
        # Initialize remedy structure
        remedy = {
            "info": "",
            "treatment": "",
            "prevention": "",
            "chemical_control": "",
            "organic_control": ""
        }
        
        # Extract information from pages
        for url in urls_to_check:
            try:
                response = requests.get(url, timeout=5)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract main content
                content = soup.get_text()
                
                # Extract disease information
                if not remedy["info"]:
                    info_patterns = [
                        re.compile(r'(?:[A-Z][^.!?]*' + re.escape(disease_name) + r'[^.!?]*\.)', re.IGNORECASE),
                        re.compile(r'(?:[A-Z][^.!?]*symptoms[^.!?]*\.)', re.IGNORECASE),
                        re.compile(r'(?:[A-Z][^.!?]*disease[^.!?]*\.)', re.IGNORECASE)
                    ]
                    for pattern in info_patterns:
                        info_matches = pattern.findall(content)
                        if info_matches:
                            remedy["info"] = '. '.join(info_matches[:2])
                            break
                
                # Extract treatment information
                if not remedy["treatment"]:
                    treatment_patterns = [
                        re.compile(r'(?:[A-Z][^.!?]*treatment[^.!?]*\.)', re.IGNORECASE),
                        re.compile(r'(?:[A-Z][^.!?]*control[^.!?]*\.)', re.IGNORECASE),
                        re.compile(r'(?:[A-Z][^.!?]*manage[^.!?]*\.)', re.IGNORECASE)
                    ]
                    for pattern in treatment_patterns:
                        treatment_matches = pattern.findall(content)
                        if treatment_matches:
                            remedy["treatment"] = '. '.join(treatment_matches[:3])
                            break
                
                # Extract prevention information
                if not remedy["prevention"]:
                    prevention_patterns = [
                        re.compile(r'(?:[A-Z][^.!?]*prevent[^.!?]*\.)', re.IGNORECASE),
                        re.compile(r'(?:[A-Z][^.!?]*avoid[^.!?]*\.)', re.IGNORECASE)
                    ]
                    for pattern in prevention_patterns:
                        prevention_matches = pattern.findall(content)
                        if prevention_matches:
                            remedy["prevention"] = '. '.join(prevention_matches[:3])
                            break
                
                # Extract chemical control information
                if not remedy["chemical_control"]:
                    chemical_patterns = [
                        re.compile(r'(?:[A-Z][^.!?]*fungicide[^.!?]*\.)', re.IGNORECASE),
                        re.compile(r'(?:[A-Z][^.!?]*pesticide[^.!?]*\.)', re.IGNORECASE),
                        re.compile(r'(?:[A-Z][^.!?]*chemical[^.!?]*control[^.!?]*\.)', re.IGNORECASE)
                    ]
                    for pattern in chemical_patterns:
                        chemical_matches = pattern.findall(content)
                        if chemical_matches:
                            remedy["chemical_control"] = '. '.join(chemical_matches[:3])
                            break
                
                # Extract organic control information
                if not remedy["organic_control"]:
                    organic_patterns = [
                        re.compile(r'(?:[A-Z][^.!?]*organic[^.!?]*\.)', re.IGNORECASE),
                        re.compile(r'(?:[A-Z][^.!?]*natural[^.!?]*\.)', re.IGNORECASE),
                        re.compile(r'(?:[A-Z][^.!?]*non.?chemical[^.!?]*\.)', re.IGNORECASE)
                    ]
                    for pattern in organic_patterns:
                        organic_matches = pattern.findall(content)
                        if organic_matches:
                            remedy["organic_control"] = '. '.join(organic_matches[:3])
                            break
                
                # If we have most of the information, break
                if all(remedy.values()):
                    break
                    
            except Exception as e:
                print(f"Error extracting from {url}: {e}")
                continue
        
        # Fill in any missing sections with default content
        if not remedy["info"]:
            remedy["info"] = f"{disease_name} is a plant disease that can affect crop health and yield."
        
        if not remedy["treatment"]:
            remedy["treatment"] = "Remove infected plant parts. Ensure proper spacing for airflow. Avoid overhead watering."
        
        if not remedy["prevention"]:
            remedy["prevention"] = "Rotate crops annually. Plant resistant varieties. Maintain good garden sanitation."
        
        if not remedy["chemical_control"]:
            remedy["chemical_control"] = "Consult with a local agricultural extension for fungicide or pesticide recommendations specific to your area."
        
        if not remedy["organic_control"]:
            remedy["organic_control"] = "Neem oil, copper-based fungicides, or horticultural oils can help control many common diseases organically."
        
        # Add a source note
        remedy["source_note"] = "This information is automatically compiled from web sources and may not be complete. Consult with local agricultural experts for specific recommendations."
        
        return remedy
        
    except Exception as e:
        print(f"Error fetching remedy for {disease_name}: {e}")
        # Return a default remedy if fetching fails
        return {
            "info": f"{disease_name} is a plant disease that requires proper management.",
            "treatment": "Remove infected plant parts. Isolate infected plants from healthy ones.",
            "prevention": "Practice crop rotation. Ensure proper plant spacing. Avoid overhead watering.",
            "chemical_control": "Consult with a local agricultural extension for specific recommendations.",
            "organic_control": "Organic options include neem oil, copper-based sprays, or horticultural oils.",
            "source_note": "Could not retrieve specific information. These are general recommendations."
        }

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save with a unique filename to avoid conflicts
    import time
    filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Process image and predict
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)[0]
    
    # Get top 3 predictions
    top_indices = predictions.argsort()[-3:][::-1]
    top_predictions = [
        {"class": class_names[i], "confidence": float(predictions[i])} 
        for i in top_indices
    ]
    
    predicted_class = class_names[np.argmax(predictions)]
    
    # Fetch remedy information dynamically
    remedy = fetch_disease_remedy(predicted_class)
    
    # Create the response with the remedy information
    result = {
        "predicted_class": predicted_class,
        "confidence": float(predictions[np.argmax(predictions)]),
        "top_predictions": top_predictions,
        "image_url": f"/uploads/{filename}",
        "remedy": remedy
    }
    
    return jsonify(result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
