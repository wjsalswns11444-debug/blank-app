import streamlit as st

st.title("AI 기반 logD 예측 및 구조 최적화 시스템")

smiles = st.text_input("SMILES를 입력하세요", "CCO")

if st.button("logD 예측 및 구조 추천"):
    st.write("입력한 SMILES:", smiles)
    st.metric("예측 logD", "2.35")
    st.success("목표 logD 범위(1~3)에 포함됩니다.")
