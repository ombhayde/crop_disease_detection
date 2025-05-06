# 🌾 Crop Disease Detection & Treatment Advisor

An AI-powered web application that detects crop diseases from images, provides dynamic treatment suggestions, and offers location-specific agricultural advice. Designed to help farmers make smarter decisions using AI and real-time data.

## 🚀 Features

- 📸 **AI Model for Disease Detection**
  - Upload crop images and get instant disease classification
  
- 🌍 **Location-Based Advice**
  - Enter your location manually to get:
    - Region-specific remedies
    - Nearby agricultural extension centers or shops
    - Climate-adapted treatment suggestions
    
- 🌱 **Dynamic Crop Metadata**
  - Gathers live information from the internet about:
    - Soil, sunlight, pH needs
    - Fertilizers and pesticides
    - Recommended resistant crop varieties
    
- 🧠 **Model Explainability**
  - Shows heatmaps (Grad-CAM) highlighting image regions that influenced the prediction
  
- 🌐 **Multilingual Support**
  - Supports Indian and global languages using Google Translate API

## 🛠️ Tech Stack

| Frontend | Backend | AI Model |
|----------|---------|----------|
| React.js + MUI + Framer Motion | Flask + Python | TensorFlow/Keras (MobileNetV2) |
| Axios (API calls) | REST APIs | Grad-CAM (Explainability) |

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/crop-disease-detection.git
cd crop-disease-detection
```

### 2. Install Frontend (React)
```bash
cd crop-disease-frontend
npm install
npm start
```

### 3. Install Backend (Flask)
```bash
cd crop-disease-backend
pip install -r requirements.txt
python app.py
```

## 📂 Project Structure
```
crop-disease-detection/
├── crop-disease-frontend/    # React frontend (MUI + Framer Motion)
├── crop-disease-backend/     # Flask backend + AI model
└── README.md
```

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /predict | POST | Detects disease from uploaded image |
| /get_crop_metadata | POST | Fetches dynamic crop info from web |
| /get_location_advice | POST | Returns remedies based on location |

## 📝 Example Prompts

**Dynamic Crop Metadata Prompt:**
"Fetch soil type, sunlight requirement, pH range, watering frequency, ideal season, fertilizer/pesticide recommendations, and resistant varieties for [crop name]. Use updated sources from the internet."

**Location-Based Advice Prompt:**
"Given the user's entered location and detected disease, recommend remedies, suggest nearby agricultural centers/shops, and tailor treatment advice based on the current climate conditions."

## 🌱 Future Enhancements

- SMS/WhatsApp alerts for treatment reminders
- Farmer community discussion board
- Multi-disease detection in a single image
- Offline PWA support for remote areas

## 🤝 Contributors

- Om — AI + Web Developer 👨‍💻
- Anjali Singh — UI/UX + Frontend 🌿

## 📄 License

Licensed under the MIT License.
Feel free to use, fork, and adapt this for research and non-commercial purposes.

## 🌟 Show Your Support!

If you found this helpful, please ⭐️ the repo and share with your friends in the farming/agri-tech community!
