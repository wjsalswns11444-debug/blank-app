import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

st.set_page_config(page_title="logD 예측 시스템", layout="centered")

st.title("AI 기반 logD 예측 및 구조 최적화 시스템")

smiles = st.text_input("SMILES를 입력하세요", "CCO")

def fake_predict_logd(smiles):
    # 지금은 테스트용 예측값
    # 나중에 여기에 GNN model.predict 코드를 연결하면 됨
    if "O" in smiles:
        return 2.35
    elif "Cl" in smiles:
        return 3.40
    else:
        return 1.80

if st.button("logD 예측 및 구조 추천"):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        st.error("올바르지 않은 SMILES입니다.")
    else:
        st.write("입력한 SMILES:", smiles)

        st.subheader("입력 분자 구조")
        img = Draw.MolToImage(mol, size=(350, 250))
        st.image(img)

        pred_logd = fake_predict_logd(smiles)

        st.subheader("예측 결과")
        st.metric("예측 logD", round(pred_logd, 3))

        if pred_logd > 3:
            st.warning("logD가 목표 범위보다 높습니다. 친수성 작용기 도입을 추천합니다.")
            st.write("추천 작용기: -OH, -NH₂, -COOH")

        elif pred_logd < 1:
            st.warning("logD가 목표 범위보다 낮습니다. 소수성 작용기 도입을 추천합니다.")
            st.write("추천 작용기: -CH₃, -Cl")

        else:
            st.success("목표 logD 범위(1~3)에 포함됩니다.")
