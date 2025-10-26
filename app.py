import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import os
from huggingface_hub import hf_hub_download

# ========================================================================
# --- 配置信息 ---
# ========================================================================
HF_REPO_ID = "Jasonzeng/EduCheck" 
# 🚨 假设模型权重文件名为 'pytorch_model.bin' 或 'model.safetensors'
# 请将其更改为你的实际模型权重文件名（例如 'training_args.bin' 如果它就是权重）
MODEL_WEIGHT_FILENAME = "pytorch_model.bin" 

# 需要下载的文件列表，确保这些文件都存在于 Jasonzeng/EduCheck 仓库根目录
REQUIRED_FILES = [
    MODEL_WEIGHT_FILENAME, 
    "config.json", 
    "tokenizer.json", 
    "special_tokens_map.json", 
    "tokenizer_config.json"
]

# --- 标签映射和 UI 配置保持不变 ---
LABEL_MAPPING = {
    0: "Non-Hallucination (Safe) ✅",
    1: "Hallucination Detected 🚨"
}
CONFIDENCE_LABELS = ["Non-Hallucination", "Hallucination"]

# ========================================================================
# --- 修复后的模型加载函数 ---
# ========================================================================

@st.cache_resource
def load_model_and_tokenizer():
    """
    手动下载模型组件并使用本地路径加载模型和分词器。
    这适用于模型文件未按标准打包，或你想要明确控制下载文件的场景。
    """
    st.info(f"正在从 Hugging Face Hub 下载和加载 {HF_REPO_ID} 的组件...")
    local_dir = "local_model_cache" # 定义一个本地缓存文件夹
    
    # 1. 下载所有必要文件到本地目录
    try:
        for filename in REQUIRED_FILES:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=local_dir,
                # 如果是私有仓库，需要添加 token=os.getenv("HF_TOKEN")
            )
        st.success(f"所有模型组件已下载到 {local_dir}")
        
    except Exception as e:
        st.error(f"❌ 模型文件下载失败。请确认 {HF_REPO_ID} 仓库中存在以下所有文件: {REQUIRED_FILES}。错误: {e}")
        return None, None

    # 2. 从本地缓存目录加载模型和分词器
    try:
        # 使用本地目录作为 from_pretrained 的路径
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForSequenceClassification.from_pretrained(local_dir)
        
        st.success("模型和分词器加载成功！")
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ 从本地文件加载模型失败。错误信息: {e}")
        st.error("请确保下载的文件是完整且正确的 Transformer 格式。")
        return None, None

# --- 在 Streamlit 应用启动时加载模型 ---
tokenizer_real, model_real = load_model_and_tokenizer()

# =======================================================
# --- 预测函数（保持不变）---
# =======================================================

def predict_hallucination(input_text: str, tokenizer, model):
    """使用加载的模型进行幻觉预测。"""
    if model is None or tokenizer is None:
        st.error("预测失败：模型未加载。请检查模型文件。")
        return None, None, None

    model.eval()
    
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

# =======================================================
# --- Streamlit UI 组件 (保持不变) ---
# =======================================================

st.set_page_config(layout="wide", page_title="EduCheck: AI Content Safety")

st.title("🛡️ EduCheck: AI 教育内容安全检测器")
st.markdown("---")

st.subheader("超越事实核查：检测教学法和概念缺陷")
st.info("该工具验证 AI 生成的教学内容的**教学健全性**和**概念准确性**。它使用 EduCheck-SFT 模型，旨在实现**“安全优先（高召回率）”**的设计目标。")

col_topic, col_answer = st.columns([1, 2])
safe_explanation = "数组是一种数据结构，用于存储固定大小的、元素类型相同且位于相邻内存位置的有序集合。"

with col_topic:
    user_topic = st.text_area(
        "1. 教学主题 (Topic):",
        "数据结构 (数组)", 
        height=100
    )

with col_answer:
    ai_answer = st.text_area(
        "2. AI 生成的解释 (Explanation):",
        safe_explanation, 
        height=100
    )

if st.button("🚨 运行 EduCheck 分析", type="primary"):
    if model_real is None:
        st.error("模型未加载。无法运行分析。请检查控制台/日志中的错误信息。")
        st.stop()
        
    if not user_topic or not ai_answer:
        st.warning("请同时填写教学主题和 AI 解释文本。")
        st.stop()

    input_text = f"### Topic:\n{user_topic}\n\n### Explanation:\n{ai_answer}"
    
    predicted_id, confidence, probabilities = predict_hallucination(
        input_text, tokenizer_real, model_real
    )
    
    if predicted_id is not None:
        st.markdown("---")
        
        col_res, col_metric = st.columns(2)
        
        predicted_label = LABEL_MAPPING.get(predicted_id)

        with col_res:
            if predicted_id == 1:
                st.error(f"### 结果: {predicted_label}")
            else:
                st.success(f"### 结果: {predicted_label}")

        with col_metric:
            st.metric("置信度分数 (Confidence)", f"{confidence:.2%}")
            
        st.markdown("#### 诊断报告")

        if predicted_id == 1:
            st.error(f"**缺陷类型:** 高风险幻觉")
            st.write(f"**诊断详情:** 模型根据其核心分类检测到高风险幻觉。**建议人工审核**以对具体的错误类型（事实性、概念性或教学法缺陷）进行分类。")
                
        else:
            st.info("模型判定内容安全。该解释**概念准确且教学法上合理**。")

        st.markdown("---")
        st.markdown("#### 详细概率分布（二元）")
        
        probs_df = pd.Series(
            probabilities, 
            index=CONFIDENCE_LABELS
        ).sort_values(ascending=False)
        
        st.bar_chart(probs_df)
