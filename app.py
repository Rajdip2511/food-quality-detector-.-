"""
Multimodal Food Quality & Safety AI - Streamlit Dashboard
PyTorch CNN + Electrochemical Sensors for Food Assessment
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
import io
import joblib
import pandas as pd
from PIL import Image as PILImage
import os

# ============================================================================
# SESSION STATE
# ============================================================================

if 'sensor_values' not in st.session_state:
    st.session_state.sensor_values = None
if 'cnn_freshness_prob' not in st.session_state:
    st.session_state.cnn_freshness_prob = None
if 'rf_safe_prob' not in st.session_state:
    st.session_state.rf_safe_prob = None
if 'rf_class' not in st.session_state:
    st.session_state.rf_class = None

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load all models"""
    models = {}
    
    # Load PyTorch CNN model
    checkpoint = torch.load('models/food_cnn_finetuned.pth', map_location='cpu')
    models['input_size'] = checkpoint.get('input_size', 224)
    models['class_to_idx'] = checkpoint.get('class_to_idx', {})
    models['idx_to_class'] = checkpoint.get('idx_to_class', {})
    
    # Define idx_to_class for class_name_from_idx function
    global idx_to_class
    idx_to_class = models['idx_to_class']
    
    # Build exact classifier architecture
    model = tv_models.mobilenet_v3_large(weights=None)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.Hardswish(),
        nn.Dropout(0.3),
        nn.Linear(256, 20),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    models['cnn_model'] = model
    
    # Load sensor models
    models['rf_model'] = joblib.load('models/sensor_rf_model.pkl')
    models['scaler'] = joblib.load('models/sensor_scaler.pkl')
    
    # Get sensor ranges from data
    df = pd.read_csv('data/mip_sensor_data.csv')
    models['sensor_ranges'] = {
        'current': (float(df['peak_current_uA'].min()), float(df['peak_current_uA'].max()), float(df['peak_current_uA'].mean())),
        'voltage': (float(df['peak_voltage_V'].min()), float(df['peak_voltage_V'].max()), float(df['peak_voltage_V'].mean())),
        'impedance': (int(df['impedance_ohm'].min()), int(df['impedance_ohm'].max()), int(df['impedance_ohm'].mean())),
        'ph': (float(df['pH_level'].min()), float(df['pH_level'].max()), float(df['pH_level'].mean()))
    }
    
    return models


def class_name_from_idx(idx, idx_to_class):
    """Convert index to display name (e.g., 5 -> 'Fresh Mango')"""
    # idx_to_class has integer keys
    class_name = idx_to_class.get(idx, f'Unknown {idx}')
    # The class_name is lowercase version of folder name (e.g., 'freshapple' from folder 'FreshApple')
    # Reconstruct proper display name by looking up folder names
    # All folders are in format: [Fresh|Rotten][Fruit|Vegetable] (e.g., FreshApple, RottenMango)
    
    # Split into prefix (fresh/rotten) and name (apple/banana/mango/etc)
    parts = class_name.lower()
    if parts.startswith('fresh'):
        prefix = 'Fresh'
        name = parts[5:]  # Remove 'fresh' prefix
    elif parts.startswith('rotten'):
        prefix = 'Rotten'
        name = parts[6:]  # Remove 'rotten' prefix
    else:
        return class_name.title()
    
    # Capitalize the name part
    name = name.capitalize()
    return f'{prefix} {name}'


def preprocess_image(img, input_size=224):
    """Preprocess image for PyTorch inference"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((input_size, input_size))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)


def predict_image(img_tensor, model, idx_to_class, top_k=1):
    """Predict image class"""
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.exp(output)
        probs, classes = torch.topk(probabilities, top_k, dim=1)
        
        results = []
        for i in range(top_k):
            class_idx = int(classes[0][i].item())
            confidence = float(probs[0][i].item())
            class_name = class_name_from_idx(class_idx, idx_to_class)
            results.append({'class': class_name, 'confidence': confidence})
        
        top_class = results[0]['class']
        top_confidence = results[0]['confidence']
        is_fresh = 'Fresh' in top_class
        freshness_prob = top_confidence if is_fresh else (1 - top_confidence)
        
        return results, top_class, top_confidence, is_fresh, float(freshness_prob)


def predict_sensor_data(scaler, rf_model, current, voltage, impedance, ph):
    """Predict sensor class"""
    features = np.array([[current, voltage, impedance, ph]])
    features_scaled = scaler.transform(features)
    prediction = int(rf_model.predict(features_scaled)[0])
    probabilities = [float(p) for p in rf_model.predict_proba(features_scaled)[0]]
    
    class_map = {0: 'Safe/Clean', 1: 'Pesticide Residue', 2: 'Spoiled/Mycotoxin'}
    return class_map.get(prediction, 'Unknown'), probabilities, prediction


def calculate_fqi(freshness_prob, safe_prob):
    """Food Quality Index = 60% visual + 40% chemical"""
    return round(float(freshness_prob * 0.6 + safe_prob * 0.4) * 100, 1)


def get_safety_banner(fqi, rf_class):
    """Get safety status banner text and color"""
    if fqi >= 85:
        return "🟢 Safe to Consume", "green"
    elif fqi >= 55:
        if rf_class == 'Pesticide Residue':
            return "🟡 Pesticide Trace Detected — Wash/Peel", "yellow"
        else:
            return "🟡 Check Freshness", "yellow"
    elif fqi >= 40:
        return "🟠 Spoiled / Do Not Consume", "orange"
    else:
        return "🔴 Spoiled / Do Not Consume", "red"


def get_dietary_guidance(is_fresh, rf_class):
    """Get single-line dietary guidance"""
    if rf_class == 'Safe/Clean':
        if is_fresh:
            return "❄️ Refrigerate (0-4°C), consume within 3-5 days"
        else:
            return "⚠️ Discard if spoilage signs present"
    elif rf_class == 'Pesticide Residue':
        return "⚠️ Wash thoroughly or peel before consuming"
    else:
        return "🚫 DO NOT CONSUME"


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="Food Quality & Safety AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.header("🍎 Food AI")
    st.markdown("---")
    
    models = load_models()
    
    st.success("✅ CNN + MIP Sensor Ready")
    st.markdown(f"**Classes**: {len(models['class_to_idx'])}")
    st.markdown(f"**Input Size**: {models['input_size']}x{models['input_size']}")
    
    st.markdown("---")
    st.subheader("Sensor Presets")
    
    ranges = models['sensor_ranges']
    
    if st.button("Clean / Safe", use_container_width=True):
        st.session_state.sensor_values = {
            'current': ranges['current'][2],
            'voltage': ranges['voltage'][2],
            'impedance': ranges['impedance'][2],
            'ph': ranges['ph'][2]
        }
        st.rerun()
    
    if st.button("Pesticide Residue Trace", use_container_width=True):
        st.session_state.sensor_values = {'current': 18.0, 'voltage': 0.55, 'impedance': 3500, 'ph': 5.8}
        st.rerun()
    
    if st.button("Spoiled / Mycotoxin", use_container_width=True):
        st.session_state.sensor_values = {'current': 12.0, 'voltage': 0.65, 'impedance': 2000, 'ph': 4.5}
        st.rerun()
    
    st.markdown("---")
    st.subheader("Sensor Parameters")
    
    if st.session_state.sensor_values is None:
        st.session_state.sensor_values = {
            'current': ranges['current'][2],
            'voltage': ranges['voltage'][2],
            'impedance': ranges['impedance'][2],
            'ph': ranges['ph'][2]
        }
    
    st.session_state.sensor_values['current'] = st.slider(
        "Peak Current (µA)", ranges['current'][0], ranges['current'][1],
        float(st.session_state.sensor_values['current']), step=0.1
    )
    st.session_state.sensor_values['voltage'] = st.slider(
        "Redox Voltage (V)", ranges['voltage'][0], ranges['voltage'][1],
        float(st.session_state.sensor_values['voltage']), step=0.01
    )
    st.session_state.sensor_values['impedance'] = st.slider(
        "Impedance (Ω)", ranges['impedance'][0], ranges['impedance'][1],
        int(st.session_state.sensor_values['impedance']), step=100
    )
    st.session_state.sensor_values['ph'] = st.slider(
        "pH Level", ranges['ph'][0], ranges['ph'][1],
        float(st.session_state.sensor_values['ph']), step=0.1
    )

# Main Content
st.title("🔬 Food Quality & Safety AI")
st.markdown("PyTorch CNN + Electrochemical Sensors")

col1, col2 = st.columns([1, 1])

# Left: Image Upload
with col1:
    st.header("📸 Image Analysis")
    uploaded_file = st.file_uploader(
        "Upload JPG, JPEG, PNG",
        type=['jpg', 'jpeg', 'png'],
        key="uploader"
    )
    
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        image = PILImage.open(io.BytesIO(image_data))
        st.image(image_data, caption="Uploaded Food", width=280)
        
        img_tensor = preprocess_image(image, models['input_size'])
        top3, top_class, top_confidence, is_fresh, freshness_prob = predict_image(
            img_tensor, models['cnn_model'], models['idx_to_class'], top_k=1
        )
        
        st.session_state.cnn_freshness_prob = float(freshness_prob)
        st.session_state.cnn_is_fresh = is_fresh
        st.session_state.cnn_confidence = float(top_confidence)
        st.session_state.cnn_top_class = top_class
        st.session_state.cnn_is_fresh = is_fresh

# Right: Results (auto-appears on upload)
with col2:
    st.header("📊 Analysis")
    
    if 'cnn_freshness_prob' in st.session_state and st.session_state.cnn_freshness_prob is not None:
        # Get sensor prediction (auto-run)
        current = float(st.session_state.sensor_values['current'])
        voltage = float(st.session_state.sensor_values['voltage'])
        impedance = float(st.session_state.sensor_values['impedance'])
        ph = float(st.session_state.sensor_values['ph'])
        
        rf_class, rf_probs, rf_pred = predict_sensor_data(
            models['scaler'], models['rf_model'],
            current, voltage, impedance, ph
        )
        
        st.session_state.rf_class = rf_class
        st.session_state.rf_safe_prob = float(rf_probs[0])
        
        # Calculate FQI
        fqi = calculate_fqi(st.session_state.cnn_freshness_prob, st.session_state.rf_safe_prob)
        
        # Big verdict banner
        verdict, color = get_safety_banner(fqi, st.session_state.rf_class)
        status = "FRESH" if st.session_state.cnn_is_fresh else "SPOILED"
        
        st.markdown(f"## **{'🟢' if status == 'FRESH' else '🔴'} {status}**")
        st.markdown(f"### **{st.session_state.cnn_top_class} — {st.session_state.cnn_confidence*100:.1f}%**")
        
        # FQI display
        st.subheader("Food Quality Index")
        st.metric("FQI", f"{fqi}/100", f"Freshness: {st.session_state.cnn_freshness_prob*100:.0f}% | Safe: {st.session_state.rf_safe_prob*100:.0f}%")
        st.progress(float(fqi / 100))
        
        # Single safety banner
        if color == "green":
            st.success(f"## {verdict}")
        elif color == "yellow":
            st.warning(f"## {verdict}")
        else:
            st.error(f"## {verdict}")
        
        # Single dietary guidance
        guidance = get_dietary_guidance(st.session_state.cnn_is_fresh, st.session_state.rf_class)
        st.info(f"**{guidance}**")
