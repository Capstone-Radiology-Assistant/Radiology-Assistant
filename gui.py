import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import cv2
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Set a custom title for the Streamlit app
st.title("LCPCXR - Localization and Categorization of Disease in Pediatric Chest X-rays")

# Sidebar for model thresholds
st.sidebar.header("Detection Settings")

# Confidence and IoU thresholds for YOLOv5
conf_thres = st.sidebar.slider("YOLOv5 Confidence Threshold", 0.0, 1.0, 0.01, 0.01)
iou_thres = st.sidebar.slider("YOLOv5 IoU Threshold", 0.0, 1.0, 0.4, 0.01)

# Define disease categories for ViT model
categories_info = [
    "Bronchitis", "Broncho-pneumonia", "Other disease", "Bronchiolitis",
    "Situs inversus", "Pneumonia", "Pleuro-pneumonia", "Diaphragmatic hernia",
    "Tuberculosis", "Congenital emphysema", "CPAM", "Hyaline membrane disease",
    "Mediastinal tumor", "Lung tumor"
]

# Load YOLOv5 model
def load_yolo_model(model_path='best.pt'):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        st.success("YOLOv5 model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        return None

# Load ViT model and apply weights if available
def load_vit_model(weights_path='vit_model_final22.pth'):
    try:
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=len(categories_info))
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # Load weights if provided
        if weights_path and os.path.isfile(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            st.success("ViT model weights loaded successfully!")
        else:
            st.warning("Using default ViT model weights.")
        
        return model, feature_extractor
    except Exception as e:
        st.error(f"Error loading ViT model: {e}")
        return None, None

# Load both models
yolo_model = load_yolo_model()
vit_model, feature_extractor = load_vit_model()

# File uploader to allow user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Convert PIL image to numpy array for YOLOv5
    img_array = np.array(image)
    
    # Run YOLOv5 inference if model is loaded
    if yolo_model:
        yolo_model.conf = conf_thres  # NMS confidence threshold
        yolo_model.iou = iou_thres    # NMS IoU threshold
        results = yolo_model(img_array)
        
        # Check if any detections were made
        if results.xyxy[0].shape[0] > 0:
            st.success("Objects detected by YOLOv5!")
            st.write("Detection Results:")
            st.write(results.pandas().xyxy[0])

            # Draw bounding boxes manually
            detections = results.xyxy[0]  # Detections for the first image
            img_with_boxes = img_array.copy()

            for *box, conf, cls in detections:
                x1, y1, x2, y2 = map(int, box)
                class_name = yolo_model.names[int(cls)]
                confidence = float(conf)
                label = f'{class_name} {confidence:.2f}'

                # Draw rectangle
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label
                cv2.putText(img_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Convert back to PIL Image for Streamlit
            result_image = Image.fromarray(img_with_boxes)
            st.image(result_image, caption="Processed Image with YOLOv5 Detections", use_container_width=True)
        else:
            st.warning("No objects detected in the image by YOLOv5.")

    # Run ViT classification if model is loaded
    if vit_model and feature_extractor:
        # Preprocess the image
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = vit_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top_prob, top_label = torch.topk(probs, k=1)

        # Display the classification result
        st.subheader("ViT Disease Classification Result")
        disease_name = categories_info[top_label[0].item()]
        confidence = top_prob[0].item() * 100
        st.write(f"Disease: **{disease_name}** with **{confidence:.2f}%** confidence.")

# Add a footer with the team name
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 18px;'>Team LCPCXR</p>", unsafe_allow_html=True)

