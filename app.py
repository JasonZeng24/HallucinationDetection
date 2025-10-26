import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd

# ========================================================================
# 🚨🚨 IMPORTANT: MODEL AND TOKENIZER LOADING SETUP 🚨🚨
# The application now strictly attempts to load the actual model.
# If model files are not present in 'checkpoint-130', the app will fail gracefully.
# ========================================================================

# --- 1. Label Mapping ---
LABEL_MAPPING = {
    0: "Non-Hallucination (Safe) ✅",
    1: "Hallucination Detected 🚨"
}
CONFIDENCE_LABELS = ["Non-Hallucination", "Hallucination"]


@st.cache_resource
def load_model_and_tokenizer():
    """
    Attempts to load the actual EduCheck-SFT model and tokenizer.
    Returns None, None if loading fails.
    """
    model_dir = "checkpoint-130" 
    model_name = "distilbert-base-uncased"
    
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the model from the saved checkpoint
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
        model.eval()
        
        st.success("✅ Model Loaded Successfully!")
        return tokenizer, model
    except Exception as e:
        # Fails gracefully if the model files are missing.
        st.error(f"❌ Error loading actual EduCheck-SFT model from '{model_dir}'. Prediction requires model files.") 
        return None, None

# --- The actual prediction function ---
def predict_hallucination(input_text: str, tokenizer, model):
    """Predicts hallucination using the loaded model."""
    if model and tokenizer:
        # Use the actual model for prediction
        inputs = tokenizer(input_text, 
                           padding="max_length", 
                           truncation=True, 
                           max_length=128, 
                           return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=1).squeeze().numpy()
        predicted_class_id = np.argmax(probabilities).item()
        confidence = probabilities[predicted_class_id].item()
        
        return predicted_class_id, confidence, probabilities
        
    else:
        # Cannot run prediction if model is None
        st.error("Prediction failed: Model not loaded. Please check model files.")
        return None, None, None

# --- Load the model (try real, fall back to failure) ---
tokenizer_real, model_real = load_model_and_tokenizer()

# =======================================================
# --- Streamlit UI Components (All English) ---
# =======================================================

st.set_page_config(layout="wide", page_title="EduCheck: AI Content Safety")

st.title("🛡️ EduCheck: AI Educational Content Safety Detector")
st.markdown("---")

st.subheader("Beyond Fact-Checking: Detecting Pedagogical & Conceptual Flaws")
st.info("This tool validates the **Pedagogical Soundness** and **Conceptual Accuracy** of AI-generated teaching content. It utilizes the EduCheck-SFT model, trained with a weighted loss function, achieving a **'Safety-First (High Recall)'** design objective.")


# --- Input Area ---
col_topic, col_answer = st.columns([1, 2])

# Set default to the SAFE case for easy demonstration of a 'Good' result
safe_explanation = "An Array is a data structure used to store a fixed-size sequential collection of elements of the same type in adjacent memory locations."

with col_topic:
    user_topic = st.text_area(
        "1. Teaching Topic (Topic):",
        "Data Structure (Array)", 
        height=100
    )

with col_answer:
    ai_answer = st.text_area(
        "2. AI-Generated Explanation (Explanation):",
        safe_explanation, 
        height=100
    )

# --- Prediction Execution ---
if st.button("🚨 Run EduCheck Analysis", type="primary"):
    if not user_topic or not ai_answer:
        st.warning("Please fill in both the Teaching Topic and the AI Explanation text.")
        st.stop()

    # Format input text
    input_text = f"### Topic:\n{user_topic}\n\n### Explanation:\n{ai_answer}"
    
    # Run Prediction
    predicted_id, confidence, probabilities = predict_hallucination(input_text, tokenizer_real, model_real)
    
    # Only proceed if prediction was successful
    if predicted_id is not None:
        st.markdown("---")
        
        # --- Result Display ---
        col_res, col_metric = st.columns(2)
        
        predicted_label = LABEL_MAPPING.get(predicted_id)

        with col_res:
            if predicted_id == 1:
                st.error(f"### Result: {predicted_label}")
            else:
                st.success(f"### Result: {predicted_label}")

        with col_metric:
            st.metric("Confidence Score (Confidence)", f"{confidence:.2%}")
            
        st.markdown("#### Diagnostic Report")

        # --- Display only general diagnosis ---
        if predicted_id == 1:
            st.error(f"**Flaw Type:** High Risk Hallucination")
            st.write(f"**Diagnosis Details:** The model detected a high-risk hallucination based on its core classification. **Human Review is Recommended** to categorize the specific error type (Factual, Conceptual, or Pedagogical).") 
                
        else:
            st.info("Model judges content as safe. This explanation is **conceptually accurate and pedagogically sound** for the target audience.")

        # --- Detailed Probability Distribution ---
        st.markdown("---")
        st.markdown("#### Detailed Probability Distribution (Binary)")
        
        probs_df = pd.Series(
            probabilities, 
            index=CONFIDENCE_LABELS
        ).sort_values(ascending=False)
        
        st.bar_chart(probs_df)
