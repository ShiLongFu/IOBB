import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('脑膨出_XGBoost_model.pkl')

# 定义特征的选项
Pupillary_dilation_options = {
    0: 'No',
    1: 'Unilateral',
    2: 'Bilateral'
}

# Define feature names
feature_names = ["Age", "Pupillary dilation","Blood glucose", "HT/MS ratio"]

# Streamlit的用户界面
st.title("Intraoperative brain bulge (IOBB) Predictor for Patients with Acute Subdural Hematoma")

# 分类选择
PD = st.selectbox("Pupillary dilation:", options=[0, 1, 2], format_func=lambda x: Pupillary_dilation_options[x])

# 数值输入
AGE = st.number_input("Age (yrs):", min_value=18, max_value=120, value=50)

# 数值输入
BG = st.number_input("Blood glucose (mmol/L):", min_value=0.01, max_value=50.00, value=7.00)

# 数值输入
HMR = st.number_input("HT/MS (hematoma thickness/midline shift) ratio:", min_value=0.01, max_value=10.00, value=1.00)

# 处理输入并进行预测
feature_values = [AGE, PD, BG, HMR]
features = np.array([feature_values])

if st.button("Predict"):
    # 预测概率（修改这里）
    predicted_proba = model.predict_proba(features)[0]
    probability_positive = predicted_proba[1] * 100  # 直接提取阳性概率

    # 显示结果（更新变量名）
    text = f"Based on feature values, predicted probability of IOBB is {probability_positive:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.tiff", bbox_inches='tight', dpi=300)
    st.image("prediction_text.tiff")

    # 使用类别1的SHAP解释（如果需要展示阳性的解释）
   # 计算SHAP值并显示力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.tiff", bbox_inches='tight', dpi=1200)
  
    st.image("shap_force_plot.tiff")
# 运行Streamlit命令生成网页应用
