# 🚀 Free Deployment Guide | বিনামূল্যে ডিপ্লয়মেন্ট গাইড

## Option 1: Hugging Face Spaces (Recommended ✅)

**সবচেয়ে সহজ এবং ভালো option - Model file সরাসরি upload করা যায়!**

### Step 1: Hugging Face Account তৈরি করুন

1. যান: https://huggingface.co/join
2. Free account তৈরি করুন

### Step 2: New Space তৈরি করুন

1. https://huggingface.co/new-space এ যান
2. **Space name:** `corn-leaf-disease-detection`
3. **License:** MIT
4. **SDK:** Gradio
5. **Hardware:** CPU Basic (Free)
6. "Create Space" ক্লিক করুন

### Step 3: Files Upload করুন

আপনার Space এ এই files upload করুন:

#### File 1: `app.py`

```
app_huggingface.py এর content copy করে app.py নামে save করুন
```

#### File 2: `requirements.txt`

```
gradio>=4.0.0
tensorflow>=2.10.0
numpy>=1.21.0
Pillow>=9.0.0
```

#### File 3: `INCEPTION_V3_CORN.h5` (আপনার model file)

- "Files" tab এ যান
- "Add file" → "Upload files" ক্লিক করুন
- আপনার .h5 model file select করুন
- Upload হতে কিছু সময় লাগবে (file size অনুযায়ী)

### Step 4: Deploy হবে automatically!

- Upload এর পর Hugging Face automatically build করবে
- 2-5 মিনিট অপেক্ষা করুন
- আপনার app live হয়ে যাবে!

### আপনার App URL:

```
https://huggingface.co/spaces/YOUR_USERNAME/corn-leaf-disease-detection
```

---

## Option 2: Google Drive + Hugging Face (বড় Model এর জন্য)

যদি model file অনেক বড় হয় (>500MB):

### Step 1: Model Google Drive এ Upload করুন

1. https://drive.google.com এ যান
2. আপনার .h5 file upload করুন
3. File এ Right click → "Share" → "Anyone with link" → "Viewer"
4. Link copy করুন

### Step 2: File ID বের করুন

Link এরকম হবে:

```
https://drive.google.com/file/d/XXXXXXXXXX/view?usp=sharing
```

`XXXXXXXXXX` হলো আপনার FILE_ID

### Step 3: app.py এ এই code ব্যবহার করুন

```python
import gdown
import os

# Download model from Google Drive
MODEL_PATH = 'INCEPTION_V3_CORN.h5'
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = 'https://drive.google.com/uc?id=YOUR_FILE_ID_HERE'
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)
```

### Step 4: requirements.txt এ add করুন

```
gdown>=4.6.0
```

---

## Option 3: Render.com (Alternative)

### Step 1: GitHub Repository তৈরি করুন

1. GitHub এ new repository তৈরি করুন
2. এই files push করুন:
   - `app.py`
   - `requirements.txt`
   - `INCEPTION_V3_CORN.h5` (Git LFS ব্যবহার করুন বড় file এর জন্য)

### Step 2: Render.com এ Deploy করুন

1. https://render.com এ free account তৈরি করুন
2. "New" → "Web Service"
3. GitHub repo connect করুন
4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
5. Deploy!

---

## 🔧 Git LFS Setup (বড় File এর জন্য)

যদি model file GitHub এ push করতে চান:

```bash
# Git LFS install
git lfs install

# Track .h5 files
git lfs track "*.h5"

# Add and commit
git add .gitattributes
git add INCEPTION_V3_CORN.h5
git commit -m "Add model file"
git push
```

---

## ❓ FAQ

### Q: Model file কত বড় হতে পারে?

- **Hugging Face Spaces:** 50GB পর্যন্ত (Git LFS)
- **Render Free:** 512MB RAM limit
- **Google Drive:** 15GB free storage

### Q: কতক্ষণ লাগে deploy হতে?

- Hugging Face: 2-5 minutes
- Render: 5-10 minutes

### Q: Free tier এ কোনো limitation আছে?

- Hugging Face: CPU only, sleep after inactivity
- Render: 750 hours/month, sleeps after 15 min inactivity

### Q: Custom domain ব্যবহার করা যাবে?

- Hugging Face: No (free tier)
- Render: Yes (paid plans)

---

## 📞 Support

যদি কোনো সমস্যা হয়, এই resources দেখুন:

- Hugging Face Docs: https://huggingface.co/docs/hub/spaces
- Gradio Docs: https://www.gradio.app/docs
- TensorFlow Docs: https://www.tensorflow.org/guide

---

**Developed by Abu Zahed (221-15-4716)**  
**Daffodil International University**
