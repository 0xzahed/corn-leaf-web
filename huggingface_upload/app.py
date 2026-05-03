"""
Corn Leaf Disease Detection - Hugging Face Spaces
By Abu Zahed (221-15-4716) | Daffodil International University
"""

import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image

# Disease classes
CLASS_NAMES = ['Common_Rust', 'Corn_Leaf_Blight', 'Gray_Leaf_Spot', 'Healthy', 'Maize_Chlorotic_Mottle_Virus']

DISPLAY_NAMES = {
    'Common_Rust': 'Common Rust (কমন রাস্ট)',
    'Corn_Leaf_Blight': 'Corn Leaf Blight (কর্ন লিফ ব্লাইট)', 
    'Gray_Leaf_Spot': 'Gray Leaf Spot (গ্রে লিফ স্পট)',
    'Healthy': 'Healthy (সুস্থ)',
    'Maize_Chlorotic_Mottle_Virus': 'MCMV (মেইজ ক্লোরোটিক মটল ভাইরাস)'
}

DISEASE_INFO = {
    'Common_Rust': ('🦠 Fungal disease by Puccinia sorghi', '💊 Apply fungicides, remove infected leaves', '⚠️ Moderate-High'),
    'Corn_Leaf_Blight': ('🍂 Caused by Exserohilum turcicum', '💊 Use resistant hybrids, foliar fungicides', '🔴 High'),
    'Gray_Leaf_Spot': ('🔘 Caused by Cercospora zeae-maydis', '💊 Plant resistant varieties, early fungicide', '⚠️ Moderate-High'),
    'Healthy': ('✅ No disease detected', '🌱 Continue good practices', '🟢 None'),
    'Maize_Chlorotic_Mottle_Virus': ('🧬 Viral disease by thrips/beetles', '💊 Remove infected plants, control vectors', '🔴 Very High')
}

# Load model
print("🌽 Loading model...")
model = load_model('INCEPTION_V3_CORN.h5')
print("✅ Model loaded!")

def analyze_image_colors(img_array):
    """
    Analyze image colors to determine if it's a corn leaf.
    Returns color metrics dictionary.
    """
    img = Image.fromarray(img_array).resize((100, 100))
    pixels = np.array(img, dtype=np.float32)
    
    r = pixels[:, :, 0]
    g = pixels[:, :, 1]
    b = pixels[:, :, 2]
    total_pixels = pixels.shape[0] * pixels.shape[1]
    
    # Green pixels (healthy corn leaf)
    green_pixels = np.sum((g > r) & (g > b) & (g > 50))
    green_ratio = green_pixels / total_pixels
    
    # Yellow-green pixels (diseased/stressed corn leaf)
    yellow_green = np.sum(
        ((g > 60) & (r > 60) & (b < g) & (g >= r * 0.7)) |
        ((g > 80) & (r > 80) & (b < 100) & (np.abs(r - g) < 50))
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
        (r < 250)
    )
    skin_ratio = skin_pixels / total_pixels
    
    # Gray/white background
    gray_pixels = np.sum(
        (np.abs(r - g) < 15) & (np.abs(g - b) < 15) & 
        ((r + g + b) > 300)
    )
    gray_ratio = gray_pixels / total_pixels
    
    # Blue (sky, water)
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

def is_corn_leaf(img_array):
    """
    Validate if image is a corn leaf based on color analysis.
    Model gives high confidence even for random images, so we rely on colors.
    """
    try:
        colors = analyze_image_colors(img_array)
        
        green = colors['green']
        yellow_green = colors['yellow_green']
        plant = colors['plant']
        skin = colors['skin']
        gray = colors['gray']
        blue = colors['blue']
        
        # Validation rules
        total_green_like = green + yellow_green
        
        # Rule 1: Must have plant colors
        if total_green_like < 0.10 and plant < 0.15:
            return False, f"No plant colors detected (green: {green:.1%}). Not a corn leaf."
        
        # Rule 2: Reject skin tones
        if skin > 0.20 and green < 0.15:
            return False, f"Human face/skin detected ({skin:.1%} skin tone)"
        
        # Rule 3: Reject gray/white background
        if gray > 0.50 and green < 0.10:
            return False, f"Mostly background ({gray:.1%} gray), not a plant"
        
        # Rule 4: Reject blue (sky/water)
        if blue > 0.30 and plant < 0.20:
            return False, f"Sky/water detected ({blue:.1%} blue), not a corn leaf"
        
        # Rule 5: Accept good green
        if green >= 0.15:
            return True, f"Valid corn leaf (green: {green:.1%})"
        
        # Rule 6: Accept yellow-green (diseased)
        if yellow_green >= 0.20 and skin < 0.15:
            return True, f"Valid corn leaf - possibly diseased (yellow-green: {yellow_green:.1%})"
        
        # Rule 7: Accept plant colors
        if plant >= 0.25 and skin < 0.20:
            return True, f"Valid plant (plant colors: {plant:.1%})"
        
        return False, f"Insufficient plant colors (green: {green:.1%})"
        
    except Exception as e:
        return False, f"Image analysis error: {str(e)}"

def predict(img):
    """Predict disease from image"""
    if img is None:
        return "⚠️ Please upload an image"
    
    try:
        # Validate if it's a corn leaf
        valid, msg = is_corn_leaf(img)
        if not valid:
            return f"❌ **Not a Corn Leaf Image**\n\n{msg}\n\n🌽 Please upload a clear image of a corn leaf for accurate disease detection."
        
        # Preprocess
        img_pil = Image.fromarray(img).convert('RGB').resize((299, 299))
        img_array = np.expand_dims(np.array(img_pil), axis=0)
        img_array = preprocess_input(img_array)
        
        # Predict
        preds = model.predict(img_array, verbose=0)[0]
        idx = np.argmax(preds)
        cls = CLASS_NAMES[idx]
        conf = float(preds[idx]) * 100
        
        desc, treat, sev = DISEASE_INFO[cls]
        
        # Build result
        result = f"""
{'='*50}
🎯 PREDICTION: {DISPLAY_NAMES[cls]}
📊 CONFIDENCE: {conf:.1f}%
{'='*50}

📋 DESCRIPTION: {desc}
💊 TREATMENT: {treat}
⚠️ SEVERITY: {sev}

{'='*50}
ALL PREDICTIONS:
"""
        for i, c in enumerate(CLASS_NAMES):
            pct = float(preds[i]) * 100
            bar = '█' * int(pct/5) + '░' * (20 - int(pct/5))
            result += f"\n{DISPLAY_NAMES[c][:25]:25} {bar} {pct:5.1f}%"
        
        result += f"""

{'='*50}
🌽 Corn Leaf Disease Detection System
Developer: Abu Zahed (221-15-4716)
Daffodil International University
"""
        return result
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Corn Leaf Image"),
    outputs=gr.Textbox(label="Analysis Result", lines=25),
    title="🌽 Corn Leaf Disease Detection",
    description="Upload a corn leaf image to detect disease. Supports: Common Rust, Corn Leaf Blight, Gray Leaf Spot, Healthy, MCMV",
    article="**Developer:** Abu Zahed (221-15-4716) | CSE, Daffodil International University",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
