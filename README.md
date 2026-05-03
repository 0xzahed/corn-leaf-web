# 🌽 Corn Leaf Disease Detection System

## Deep Learning Based Corn Leaf Disease Classification Using CNN Architecture

**By Abu Zahed (221-15-4716)**  
Department of Computer Science and Engineering  
Daffodil International University

---

## 📋 Project Overview

This web application uses deep learning (InceptionV3 CNN architecture) to detect and classify corn leaf diseases from uploaded images. The system can identify 5 classes:

1. **Common Rust** - Fungal disease caused by Puccinia sorghi
2. **Corn Leaf Blight** - Caused by Exserohilum turcicum
3. **Gray Leaf Spot** - Caused by Cercospora zeae-maydis
4. **Healthy** - No disease detected
5. **Maize Chlorotic Mottle Virus (MCMV)** - Viral disease

## 🚀 Features

- ✅ Upload corn leaf images via drag & drop or file browser
- ✅ Real-time disease detection and classification
- ✅ Confidence score for predictions
- ✅ Probability distribution for all disease classes
- ✅ Bilingual support (English & Bengali)
- ✅ Disease information and treatment suggestions
- ✅ Detection of non-corn leaf images

## 📁 Project Structure

```
corn-leaf-webapp/
├── app.py                    # Flask application
├── INCEPTION_V3_CORN.h5      # Trained model file
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── templates/
│   └── index.html           # Web interface
└── uploads/                  # Temporary upload folder
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone/Navigate to Project Directory

```bash
cd /media/panda/Data1/corn-leaf-webapp
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python app.py
```

### Step 5: Access the Web Application

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

## 📸 How to Use

1. **Open the web application** in your browser
2. **Upload a corn leaf image** by:
   - Dragging and dropping the image onto the upload zone
   - Clicking the upload zone to browse files
3. **Click "Detect Disease"** button
4. **View the results** including:
   - Predicted disease class
   - Confidence percentage
   - Disease information in English & Bengali
   - Treatment recommendations
   - All class probabilities

## ⚠️ Important Notes

- **Only corn leaf images** are supported
- Non-corn images will be detected and rejected
- Supported formats: **JPG, JPEG, PNG**
- Maximum file size: **16MB**
- For best results, use clear, well-lit images of corn leaves

## 🔬 Model Information

- **Architecture**: InceptionV3 (Transfer Learning)
- **Input Size**: 224 x 224 pixels
- **Training Dataset**: 7,068 augmented images from Shahjahanpur, Bogura, Bangladesh
- **Accuracy**: ~99.30%

## 👨‍🏫 Supervisors

- **Mr. Mayen Uddin Mojumdar** (Assistant Professor, CSE, DIU)
- **Dr. Md. Ali Hossain** (Associate Professor, CSE, DIU)

## 📞 Contact

**Abu Zahed**  
Student ID: 221-15-4716  
Department of Computer Science and Engineering  
Daffodil International University  
Dhaka, Bangladesh

---

© 2026 Daffodil International University. All Rights Reserved.
# corn-leaf-web
