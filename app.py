"""
Corn Leaf Disease Detection Web Application
Deep Learning Based Corn Leaf Disease Classification Using CNN Architecture
By Abu Zahed (221-15-4716)
Daffodil International University

Supervised by: Mr. Mayen Uddin Mojumdar
Co-Supervised by: Dr. Md. Ali Hossain
"""

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Disease classes - as per the report (Keras sorts alphabetically by folder name)
# Verify this matches your training data folder order
CLASS_NAMES = [
    'Common_Rust',
    'Corn_Leaf_Blight', 
    'Gray_Leaf_Spot',
    'Healthy',
    'Maize_Chlorotic_Mottle_Virus'
]

# Disease information in Bengali and English
DISEASE_INFO = {
    'Common Rust': {
        'en': 'Common Rust is a fungal disease caused by Puccinia sorghi. It appears as small, circular to elongated, cinnamon-brown pustules on both leaf surfaces.',
        'bn': 'কমন রাস্ট একটি ছত্রাকজনিত রোগ যা Puccinia sorghi দ্বারা সৃষ্ট। এটি পাতার উভয় পৃষ্ঠে ছোট, বৃত্তাকার থেকে লম্বাটে, দারুচিনি-বাদামী রঙের পুস্টুল হিসেবে দেখা যায়।',
        'treatment': 'Apply fungicides containing azoxystrobin or propiconazole. Remove infected leaves and improve air circulation.',
        'severity': 'Moderate to High'
    },
    'Corn Leaf Blight': {
        'en': 'Northern Corn Leaf Blight (NCLB) is caused by the fungus Exserohilum turcicum. It produces long, elliptical, grayish-green or tan lesions on leaves.',
        'bn': 'নর্দার্ন কর্ন লিফ ব্লাইট (NCLB) Exserohilum turcicum ছত্রাক দ্বারা সৃষ্ট। এটি পাতায় লম্বা, উপবৃত্তাকার, ধূসর-সবুজ বা ট্যান রঙের ক্ষত তৈরি করে।',
        'treatment': 'Use resistant hybrids, apply foliar fungicides, practice crop rotation.',
        'severity': 'High'
    },
    'Gray Leaf Spot': {
        'en': 'Gray Leaf Spot is caused by Cercospora zeae-maydis. It appears as rectangular, tan to gray lesions that run parallel to leaf veins.',
        'bn': 'গ্রে লিফ স্পট Cercospora zeae-maydis দ্বারা সৃষ্ট। এটি আয়তাকার, ট্যান থেকে ধূসর ক্ষত হিসেবে দেখা যায় যা পাতার শিরার সমান্তরালে চলে।',
        'treatment': 'Plant resistant varieties, apply fungicides early, reduce crop residue.',
        'severity': 'Moderate to High'
    },
    'Healthy': {
        'en': 'The corn leaf is healthy with no visible signs of disease. Continue regular monitoring and maintenance.',
        'bn': 'ভুট্টার পাতা সুস্থ এবং কোনো রোগের লক্ষণ নেই। নিয়মিত পর্যবেক্ষণ এবং রক্ষণাবেক্ষণ চালিয়ে যান।',
        'treatment': 'No treatment needed. Maintain good agricultural practices.',
        'severity': 'None'
    },
    'Maize Chlorotic Mottle Virus': {
        'en': 'Maize Chlorotic Mottle Virus (MCMV) causes mottling, chlorosis, and stunting. It is transmitted by thrips and beetles.',
        'bn': 'মেইজ ক্লোরোটিক মটল ভাইরাস (MCMV) মটলিং, ক্লোরোসিস এবং স্টান্টিং সৃষ্টি করে। এটি থ্রিপস এবং বিটল দ্বারা সংক্রমিত হয়।',
        'treatment': 'Remove infected plants, control insect vectors, use virus-free seeds.',
        'severity': 'Very High'
    }
}

# Load the model
MODEL_PATH = 'INCEPTION_V3_CORN.h5'
model = None

def load_disease_model():
    """Load the trained model"""
    global model
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(299, 299)):
    """Preprocess image for model prediction - InceptionV3 requires 299x299"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Use InceptionV3's preprocessing (scales pixels to -1 to 1 range)
    img_array = preprocess_input(img_array)
    return img_array

def calculate_prediction_entropy(predictions):
    """
    Calculate entropy of predictions to measure uncertainty.
    High entropy = model is uncertain (similar probabilities for all classes)
    Low entropy = model is confident (one class has high probability)
    """
    epsilon = 1e-10
    probs = np.clip(predictions, epsilon, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(predictions))
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

def analyze_image_colors(img_path):
    """
    Analyze image colors to determine if it's a corn leaf.
    Returns color metrics dictionary.
    """
    img = Image.open(img_path)
    img_rgb = img.convert('RGB')
    img_small = img_rgb.resize((100, 100))
    pixels = np.array(img_small)
    
    r = pixels[:, :, 0].astype(float)
    g = pixels[:, :, 1].astype(float)
    b = pixels[:, :, 2].astype(float)
    total_pixels = pixels.shape[0] * pixels.shape[1]
    
    # Green pixels (healthy corn leaf)
    green_pixels = np.sum((g > r) & (g > b) & (g > 50))
    green_ratio = green_pixels / total_pixels
    
    # Yellow-green pixels (diseased/stressed corn leaf)
    yellow_green = np.sum(
        ((g > 60) & (r > 60) & (b < g) & (g >= r * 0.7)) |  # Yellow-green
        ((g > 80) & (r > 80) & (b < 100) & (np.abs(r - g) < 50))  # Brownish-green
    )
    yellow_green_ratio = yellow_green / total_pixels
    
    # Brown/tan pixels (diseased areas, rust)
    brown_pixels = np.sum(
        (r > 80) & (g > 50) & (b < 80) &
        (r > b) & (g > b) &
        (r < 200) & (np.abs(r - g) < 60)
    )
    brown_ratio = brown_pixels / total_pixels
    
    # Total plant-like colors
    plant_ratio = min(1.0, green_ratio + yellow_green_ratio * 0.7 + brown_ratio * 0.3)
    
    # Skin tone detection
    skin_pixels = np.sum(
        (r > 95) & (g > 40) & (b > 20) &
        (r > g) & (r > b) &
        ((r - g) > 15) & ((r - b) > 15) &
        (r < 250)  # Not pure white
    )
    skin_ratio = skin_pixels / total_pixels
    
    # Gray/white background
    gray_pixels = np.sum(
        (np.abs(r - g) < 15) & (np.abs(g - b) < 15) & 
        ((r + g + b) > 300)  # Light gray/white
    )
    gray_ratio = gray_pixels / total_pixels
    
    # Blue (sky, water - not plant)
    blue_pixels = np.sum((b > r * 1.2) & (b > g * 1.2) & (b > 80))
    blue_ratio = blue_pixels / total_pixels
    
    return {
        'green': green_ratio,
        'yellow_green': yellow_green_ratio,
        'brown': brown_ratio,
        'plant': plant_ratio,
        'skin': skin_ratio,
        'gray': gray_ratio,
        'blue': blue_ratio
    }

def is_likely_corn_leaf(img_path, predictions):
    """
    Validate if image is a corn leaf based on:
    1. Color analysis (must have plant-like colors)
    2. Model confidence patterns
    
    Model behavior note: This model gives high confidence even for random images,
    so we MUST rely primarily on color analysis for validation.
    """
    try:
        # ===== COLOR ANALYSIS (PRIMARY VALIDATION) =====
        colors = analyze_image_colors(img_path)
        
        green = colors['green']
        yellow_green = colors['yellow_green']
        plant = colors['plant']
        skin = colors['skin']
        gray = colors['gray']
        blue = colors['blue']
        
        # ===== VALIDATION RULES =====
        
        # Rule 1: Must have meaningful plant colors
        # Corn leaves should have green or yellow-green colors
        total_green_like = green + yellow_green
        
        if total_green_like < 0.10 and plant < 0.15:
            return False, f"No plant colors detected (green: {green:.1%}, plant: {plant:.1%}). This is not a corn leaf."
        
        # Rule 2: Reject if dominated by skin tones with low green
        if skin > 0.20 and green < 0.15:
            return False, f"Image appears to be a person ({skin:.1%} skin tone, only {green:.1%} green)"
        
        # Rule 3: Reject if mostly gray/white background with no green
        if gray > 0.50 and green < 0.10:
            return False, f"Image is mostly background ({gray:.1%} gray) with minimal plant content"
        
        # Rule 4: Reject if too blue (sky, water)
        if blue > 0.30 and plant < 0.20:
            return False, f"Image appears to be sky/water ({blue:.1%} blue), not a corn leaf"
        
        # Rule 5: For valid corn leaf, require reasonable green content
        # Even diseased leaves should have some green remaining
        if green >= 0.15:
            return True, f"Valid corn leaf (green: {green:.1%})"
        
        # Rule 6: Accept yellow-green images (likely diseased but still corn)
        if yellow_green >= 0.20 and skin < 0.15:
            return True, f"Valid corn leaf - possibly diseased (yellow-green: {yellow_green:.1%})"
        
        # Rule 7: If some plant colors exist and not clearly rejected, accept
        if plant >= 0.25 and skin < 0.20:
            return True, f"Valid plant image (plant colors: {plant:.1%})"
        
        # Default: reject if we're unsure
        return False, f"Insufficient plant colors (green: {green:.1%}, plant: {plant:.1%})"
        
    except Exception as e:
        # If color analysis fails, be conservative
        return False, f"Image analysis error: {str(e)}"

def predict_disease(img_path):
    """Make prediction on the image"""
    global model
    
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Preprocess image
        img_array = preprocess_image(img_path)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predictions = predictions[0]
        
        # Check if it's likely a corn leaf (pass image path for color analysis)
        is_corn, message = is_likely_corn_leaf(img_path, predictions)
        
        if not is_corn:
            # Calculate additional debug info
            entropy = calculate_prediction_entropy(predictions)
            return {
                'is_corn_leaf': False,
                'message': 'এই ছবিটি ভুট্টার পাতার মতো মনে হচ্ছে না। অনুগ্রহ করে একটি পরিষ্কার ভুট্টার পাতার ছবি আপলোড করুন।',
                'message_en': f'This image does not appear to be a corn leaf. {message}. Please upload a clear corn leaf image.',
                'debug_info': {
                    'max_confidence': float(np.max(predictions)) * 100,
                    'entropy': float(entropy),
                    'reason': message,
                    'all_predictions': {
                        CLASS_NAMES[i]: float(predictions[i]) * 100 
                        for i in range(len(CLASS_NAMES))
                    }
                }
            }, None
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx]) * 100
        
        # Get all class probabilities
        all_predictions = {
            CLASS_NAMES[i]: float(predictions[i]) * 100 
            for i in range(len(CLASS_NAMES))
        }
        
        # Sort predictions by confidence
        sorted_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
        
        result = {
            'is_corn_leaf': True,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'all_predictions': sorted_predictions,
            'disease_info': DISEASE_INFO.get(predicted_class, {}),
            'is_healthy': predicted_class == 'Healthy'
        }
        
        return result, None
        
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Render main page"""
    model_loaded = model is not None
    return render_template('index.html', model_loaded=model_loaded, classes=CLASS_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload JPG, JPEG, or PNG images.'}), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result, error = predict_disease(filepath)
        
        # Clean up - remove uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'classes': CLASS_NAMES
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🌽 Corn Leaf Disease Detection System")
    print("=" * 60)
    print("By Abu Zahed (221-15-4716)")
    print("Daffodil International University")
    print("=" * 60)
    
    # Load model on startup
    if load_disease_model():
        print("🚀 Starting web server...")
        print("📍 Open http://127.0.0.1:5000 in your browser")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please check the model file path.")
