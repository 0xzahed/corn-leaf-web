"""
Corn Leaf Disease Detection - Hugging Face Spaces Deployment
Deep Learning Based Corn Leaf Disease Classification Using CNN Architecture
By Abu Zahed (221-15-4716)
Daffodil International University
"""

import os
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image

# Disease classes
CLASS_NAMES = [
    'Common_Rust',
    'Corn_Leaf_Blight', 
    'Gray_Leaf_Spot',
    'Healthy',
    'Maize_Chlorotic_Mottle_Virus'
]

# Display names (for showing to users)
DISPLAY_NAMES = {
    'Common_Rust': 'Common Rust (কমন রাস্ট)',
    'Corn_Leaf_Blight': 'Corn Leaf Blight (কর্ন লিফ ব্লাইট)', 
    'Gray_Leaf_Spot': 'Gray Leaf Spot (গ্রে লিফ স্পট)',
    'Healthy': 'Healthy (সুস্থ)',
    'Maize_Chlorotic_Mottle_Virus': 'Maize Chlorotic Mottle Virus (MCMV)'
}

# Disease information
DISEASE_INFO = {
    'Common_Rust': {
        'description': '🦠 Common Rust is a fungal disease caused by Puccinia sorghi.',
        'description_bn': 'কমন রাস্ট একটি ছত্রাকজনিত রোগ যা Puccinia sorghi দ্বারা সৃষ্ট।',
        'treatment': '💊 Apply fungicides containing azoxystrobin or propiconazole. Remove infected leaves.',
        'severity': '⚠️ Moderate to High'
    },
    'Corn_Leaf_Blight': {
        'description': '🍂 Northern Corn Leaf Blight (NCLB) is caused by the fungus Exserohilum turcicum.',
        'description_bn': 'নর্দার্ন কর্ন লিফ ব্লাইট Exserohilum turcicum ছত্রাক দ্বারা সৃষ্ট।',
        'treatment': '💊 Use resistant hybrids, apply foliar fungicides, practice crop rotation.',
        'severity': '🔴 High'
    },
    'Gray_Leaf_Spot': {
        'description': '🔘 Gray Leaf Spot is caused by Cercospora zeae-maydis.',
        'description_bn': 'গ্রে লিফ স্পট Cercospora zeae-maydis দ্বারা সৃষ্ট।',
        'treatment': '💊 Plant resistant varieties, apply fungicides early, reduce crop residue.',
        'severity': '⚠️ Moderate to High'
    },
    'Healthy': {
        'description': '✅ The corn leaf is healthy with no visible signs of disease.',
        'description_bn': 'ভুট্টার পাতা সুস্থ এবং কোনো রোগের লক্ষণ নেই।',
        'treatment': '🌱 No treatment needed. Maintain good agricultural practices.',
        'severity': '🟢 None'
    },
    'Maize_Chlorotic_Mottle_Virus': {
        'description': '🧬 Maize Chlorotic Mottle Virus (MCMV) causes mottling and stunting.',
        'description_bn': 'মেইজ ক্লোরোটিক মটল ভাইরাস (MCMV) মটলিং এবং স্টান্টিং সৃষ্টি করে।',
        'treatment': '💊 Remove infected plants, control insect vectors, use virus-free seeds.',
        'severity': '🔴 Very High'
    }
}

# Load model
print("Loading model...")
model = load_model('INCEPTION_V3_CORN.h5')
print("✅ Model loaded successfully!")

def is_corn_leaf(img_pil, predictions):
    """Check if image is a corn leaf using color analysis"""
    try:
        img_small = img_pil.resize((150, 150))
        pixels = np.array(img_small, dtype=np.float32)
        
        r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
        total_pixels = pixels.shape[0] * pixels.shape[1]
        
        # Skin detection
        skin_mask = (
            (r > 95) & (g > 40) & (b > 20) &
            (r > g) & (r > b) &
            (np.abs(r - g) > 15)
        )
        skin_ratio = np.sum(skin_mask) / total_pixels
        
        if skin_ratio > 0.25:
            return False, "এটি মানুষের ছবি মনে হচ্ছে, ভুট্টার পাতা নয়।"
        
        # Plant color detection
        green_mask = (g > r * 0.9) & (g > b) & (g > 60)
        yellow_green_mask = (g > 70) & (r > 70) & (g >= r * 0.7) & (b < g * 0.8)
        
        plant_ratio = (np.sum(green_mask) + np.sum(yellow_green_mask) * 0.7) / total_pixels
        
        if plant_ratio < 0.15:
            return False, f"এটি ভুট্টার পাতার ছবি মনে হচ্ছে না। উদ্ভিদের রং পাওয়া যায়নি।"
        
        return True, "Valid"
        
    except:
        return True, "Check skipped"

def predict_disease(input_image):
    """Main prediction function for Gradio"""
    if input_image is None:
        return "❌ অনুগ্রহ করে একটি ছবি আপলোড করুন।", None
    
    try:
        # Convert to PIL if numpy array
        if isinstance(input_image, np.ndarray):
            img_pil = Image.fromarray(input_image).convert('RGB')
        else:
            img_pil = input_image.convert('RGB')
        
        # Preprocess for model
        img_resized = img_pil.resize((299, 299))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Check if corn leaf
        is_valid, message = is_corn_leaf(img_pil, predictions)
        
        if not is_valid:
            return f"""
## ❌ এটি ভুট্টার পাতা নয় | Not a Corn Leaf

{message}

This image does not appear to be a corn leaf. Please upload a clear image of a corn leaf.

---
**অনুগ্রহ করে একটি ভুট্টার পাতার ছবি আপলোড করুন।**
""", None
        
        # Get prediction
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100
        
        # Get disease info
        info = DISEASE_INFO.get(predicted_class, {})
        display_name = DISPLAY_NAMES.get(predicted_class, predicted_class)
        
        # Create confidence dict for label output
        confidence_dict = {
            DISPLAY_NAMES[CLASS_NAMES[i]]: float(predictions[i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        # Build result text
        is_healthy = predicted_class == 'Healthy'
        status_emoji = "✅" if is_healthy else "⚠️"
        status_text = "সুস্থ | Healthy" if is_healthy else "রোগ সনাক্ত হয়েছে | Disease Detected"
        
        result_text = f"""
## {status_emoji} {status_text}

### 🎯 Prediction: **{display_name}**
### 📊 Confidence: **{confidence:.2f}%**

---

### 📋 Disease Information | রোগের তথ্য

**English:** {info.get('description', 'N/A')}

**বাংলা:** {info.get('description_bn', 'N/A')}

---

### 💊 Treatment | চিকিৎসা
{info.get('treatment', 'N/A')}

### ⚡ Severity | তীব্রতা
{info.get('severity', 'N/A')}

---
*Developed by Abu Zahed (221-15-4716) | Daffodil International University*
"""
        
        return result_text, confidence_dict
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# Create Gradio Interface
with gr.Blocks(
    title="🌽 Corn Leaf Disease Detection",
    theme=gr.themes.Soft(primary_hue="green")
) as demo:
    
    gr.Markdown("""
    # 🌽 ভুট্টার পাতার রোগ সনাক্তকরণ সিস্টেম
    # Corn Leaf Disease Detection System
    
    **Deep Learning Based Classification Using InceptionV3 CNN Architecture**
    
    গভীর শিক্ষণ ভিত্তিক শ্রেণীবিভাগ | InceptionV3 CNN আর্কিটেকচার ব্যবহার করে
    
    ---
    
    ### 📤 Upload a corn leaf image to detect disease | রোগ সনাক্ত করতে ভুট্টার পাতার ছবি আপলোড করুন
    
    **Supported Classes | সমর্থিত শ্রেণী:**
    - 🦠 Common Rust (কমন রাস্ট)
    - 🍂 Corn Leaf Blight (কর্ন লিফ ব্লাইট)
    - 🔘 Gray Leaf Spot (গ্রে লিফ স্পট)
    - 🌿 Healthy (সুস্থ)
    - 🧬 Maize Chlorotic Mottle Virus (MCMV)
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📷 Upload Corn Leaf Image | ভুট্টার পাতার ছবি আপলোড করুন",
                type="pil",
                height=350
            )
            predict_btn = gr.Button(
                "🔍 Detect Disease | রোগ সনাক্ত করুন", 
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            output_text = gr.Markdown(
                label="Result | ফলাফল",
                value="*Upload an image and click 'Detect Disease' to see results.*"
            )
            output_label = gr.Label(
                label="📊 Confidence Scores | আত্মবিশ্বাস স্কোর",
                num_top_classes=5
            )
    
    predict_btn.click(
        fn=predict_disease,
        inputs=input_image,
        outputs=[output_text, output_label]
    )
    
    gr.Markdown("""
    ---
    
    ### ℹ️ Important Notes | গুরুত্বপূর্ণ নোট
    
    - ⚠️ **Only corn leaf images** are supported | শুধুমাত্র ভুট্টার পাতার ছবি সমর্থিত
    - 📸 Use clear, well-lit images | স্পষ্ট, ভালো আলোর ছবি ব্যবহার করুন
    - 🚫 Human/face images will be rejected | মানুষের ছবি প্রত্যাখ্যান করা হবে
    
    ---
    
    ### 👨‍💻 Developer Information
    
    **Abu Zahed** (221-15-4716)  
    Department of Computer Science and Engineering  
    Daffodil International University
    
    **Supervised by:** Mr. Mayen Uddin Mojumdar (Assistant Professor)  
    **Co-Supervised by:** Dr. Md. Ali Hossain (Associate Professor)
    
    ---
    
    © 2026 Daffodil International University. All Rights Reserved.
    """)

# Launch
if __name__ == "__main__":
    demo.launch()
