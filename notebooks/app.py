import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps
import numpy as np
import time

# --- NEW IMPORTS FOR EXPLAINABILITY ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================================
# 1. CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="CarDefect Sentinel",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #00c6ff, #0072ff);
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #41444e;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. UTILS: PREPROCESSING & CAM
# ==========================================
class LetterboxResize:
    def __init__(self, size, fill_color=(0, 0, 0)):
        self.size = size
        self.fill_color = fill_color

    def __call__(self, img):
        target_w, target_h = self.size, self.size
        w, h = img.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        return ImageOps.expand(img, padding, fill=self.fill_color)

def get_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def generate_gradcam(model, input_tensor, rgb_img_array, target_class_idx):
    """
    Generates a heatmap for a specific class index.
    """
    # 1. Target the last convolutional layer of ConvNeXt-Tiny
    target_layers = [model.features[-1][-1]]

    # 2. Initialize CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # 3. Define the specific class we want to explain
    targets = [ClassifierOutputTarget(target_class_idx)]

    # 4. Generate
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # 5. Overlay on original image
    # Note: rgb_img_array should be float32 in [0, 1] for this function
    visualization = show_cam_on_image(rgb_img_array / 255.0, grayscale_cam, use_rgb=True)
    return visualization

# ==========================================
# 3. MODEL LOADING
# ==========================================
@st.cache_resource
def load_model(model_path="results/final_finetuned_model.pth", num_classes=4, device="cpu"):
    try:
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# ==========================================
# 4. MAIN UI LOGIC
# ==========================================
st.title("🚗 CarDefect Sentinel // V1.0")

with st.sidebar:
    st.header("⚙️ Control Panel")
    confidence_threshold = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    upload_file = st.file_uploader("Upload Inspection Image", type=['jpg', 'png', 'jpeg'])

if upload_file is not None:
    # --- A. PREPARE IMAGE ---
    image = Image.open(upload_file).convert('RGB')
    
    # Create the display version (resized to 224x224 for Grad-CAM overlay)
    vis_img = image.resize((224, 224), Image.Resampling.BICUBIC)
    vis_img_array = np.array(vis_img)
    
    # Prepare Tensor
    transform = get_transforms(224)
    input_tensor = transform(image).unsqueeze(0)
    
    # --- B. INFERENCE ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)
    
    # Stop execution if model didn't load (Fixes your NameError)
    if model is None:
        st.stop()

    with torch.no_grad():
        output = model(input_tensor.to(device))
        probs = torch.sigmoid(output).squeeze().cpu().numpy()

    classes = ['Broken Glass', 'Dent', 'Scratch', 'Wreck']
    
    # --- C. DISPLAY RESULTS ---
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("🔍 Visual Analysis")
        st.image(image, use_container_width=True, caption="Original Input")
        
        # --- NEW: GRAD-CAM SECTION ---
        # Find indices where confidence > threshold
        active_indices = [i for i, p in enumerate(probs) if p > confidence_threshold]
        
        if active_indices:
            st.divider()
            st.markdown("#### 🧠 Model Explainability (Grad-CAM)")
            st.caption("Heatmaps show where the model is 'looking' to make its decision.")
            
            # Create columns for heatmaps (max 3 per row)
            cam_cols = st.columns(len(active_indices))
            
            for idx, col in zip(active_indices, cam_cols):
                class_name = classes[idx]
                
                # Generate CAM for this specific class
                # Note: We must enable gradients for CAM, even in inference mode
                with torch.set_grad_enabled(True):
                    heatmap = generate_gradcam(model, input_tensor.to(device), vis_img_array, idx)
                
                with col:
                    st.image(heatmap, caption=f"Focus: {class_name}", use_container_width=True)

    with col2:
        st.subheader("📊 Damage Report")
        
        detected = [classes[i] for i in active_indices]
        
        if detected:
            st.error(f"VERDICT: {len(detected)} DEFECTS DETECTED")
        else:
            st.success("VERDICT: CLEAN PASS")

        st.markdown("---")
        
        for i, class_name in enumerate(classes):
            score = float(probs[i])
            percent = int(score * 100)
            color = "#ff4b4b" if score > confidence_threshold else "#00c853"
            
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: bold; color: #eee;">{class_name}</span>
                    <span style="color: {color}; font-weight: bold;">{percent}%</span>
                </div>
                <div style="width: 100%; background-color: #444; border-radius: 5px; height: 8px;">
                    <div style="width: {percent}%; background-color: {color}; height: 8px; border-radius: 5px;"></div>
                </div>
            </div>
            <br>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 50px; color: #666;">
        <h2>Waiting for input stream...</h2>
    </div>
    """, unsafe_allow_html=True)