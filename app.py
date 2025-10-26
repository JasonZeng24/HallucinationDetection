import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import os

# ========================================================================
# --- 配置信息 ---
# ========================================================================
# 🚨 你的 Hugging Face 仓库 ID
HF_REPO_ID = "Jasonzeng/EduCheck" 

# --- 标签映射和 UI 配置 ---
LABEL_MAPPING = {
    0: "Non-Hallucination (Safe) ✅",
    1: "Hallucination Detected 🚨"
}
CONFIDENCE_LABELS = ["Non-Hallucination", "Hallucination"]

# ========================================================================
# --- 模型加载函数 (使用 Streamlit 缓存) ---
# ========================================================================

@st.cache_resource
def load_model_and_tokenizer():
    """
    加载模型和分词器。使用 @st.cache_resource 确保只在 Streamlit 启动时运行一次。
    如果 HF 仓库配置正确，from_pretrained() 会自动处理大文件的下载和缓存。
    """
    st.info(f"正在从 Hugging Face Hub 加载模型: {HF_REPO_ID}...")
    try:
        # 使用仓库 ID 直接加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
        model = AutoModelForSequenceClassification.from_pretrained(HF_REPO_ID)
        
        st.success("模型和分词器加载成功！")
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ 加载模型或分词器失败。错误信息: {e}")
        st.error("请立即检查你的 Hugging Face 仓库，确保所有必要的配置文件（config.json, tokenizer.json 等）都已上传。")
        return None, None

# --- 在 Streamlit 应用启动时加载模型 ---
tokenizer_real, model_real = load_model_and_tokenizer()

# ========================================================================
# --- 预测函数 ---
# ========================================================================

def predict_hallucination(input_text: str, tokenizer, model):
    """使用加载的模型进行幻觉预测。"""
    if model is None or tokenizer is None:
        return None, None, None

    # 设置模型为评估模式
    model.eval()
    
    # 编码输入文本
    inputs = tokenizer(input_text, 
                       padding="max_length", 
                       truncation=True, 
                       max_length=128, 
                       return_tensors="pt")
    
    # 禁用梯度计算
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 计算概率
    probabilities = torch.softmax(outputs.logits, dim=1).squeeze().numpy()
    
    # 获取预测类别ID和置信度
    predicted_class_id = np.argmax(probabilities).item()
    confidence = probabilities[predicted_class_id].item()
    
    return predicted_class_id, confidence, probabilities

# =======================================================
# --- Streamlit UI 组件 ---
# =======================================================

st.set_page_config(layout="wide", page_title="EduCheck: AI Content Safety")

st.title("🛡️ EduCheck: AI 教育内容安全检测器")
st.markdown("---")

st.subheader("超越事实核查：检测教学法和概念缺陷")
st.info("该工具验证 AI 生成的教学内容的**教学健全性**和**概念准确性**。")

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
        st.error("模型未加载。无法运行分析。请检查 Streamlit 日志中的错误信息。")
        st.stop()
        
    if not user_topic or not ai_answer:
        st.warning("请同时填写教学主题和 AI 解释文本。")
        st.stop()

    # 格式化输入文本
    input_text = f"### Topic:\n{user_topic}\n\n### Explanation:\n{ai_answer}"
    
    # 运行预测
    predicted_id, confidence, probabilities = predict_hallucination(
        input_text, tokenizer_real, model_real
    )
    
    # 仅在预测成功时继续
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

        # --- 显示诊断详情 ---
        if predicted_id == 1:
            st.error(f"**缺陷类型:** 高风险幻觉")
            st.write(f"**诊断详情:** 模型检测到高风险幻觉。**建议人工审核**。")
                
        else:
            st.info("模型判定内容安全。该解释**概念准确且教学法上合理**。")

        # --- 详细概率分布 ---
        st.markdown("---")
        st.markdown("#### 详细概率分布（二元）")
        
        probs_df = pd.Series(
            probabilities, 
            index=CONFIDENCE_LABELS
        ).sort_values(ascending=False)
        
        st.bar_chart(probs_df)
