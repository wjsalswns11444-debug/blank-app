import streamlit as st
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw, AllChem

import tensorflow as tf
from tensorflow.keras import layers


# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="logD 예측 및 구조 최적화", layout="wide")

MODEL_PATH = "gnn_logd_model.keras"
TARGET_MIN = 1.0
TARGET_MAX = 3.0

ATOM_LIST = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
MAX_ATOMS = 115


# =========================
# GNN Custom Layer
# =========================
class GraphConv(layers.Layer):
    def __init__(self, units, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.dense = layers.Dense(units, activation=activation)

    def call(self, inputs):
        X, A = inputs
        H = tf.matmul(A, X)
        return self.dense(H)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": self.activation
        })
        return config


class MaskedMeanPooling(layers.Layer):
    def call(self, inputs):
        H, X_input = inputs
        mask = tf.cast(
            tf.reduce_sum(tf.abs(X_input), axis=-1, keepdims=True) > 0,
            tf.float32
        )
        H = H * mask
        return tf.reduce_sum(H, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-8)


# =========================
# 모델 불러오기
# =========================
@st.cache_resource
def load_gnn_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "GraphConv": GraphConv,
            "MaskedMeanPooling": MaskedMeanPooling
        },
        compile=False
    )
    return model


# =========================
# SMILES → GNN 입력 변환
# =========================
def atom_features(atom):
    atomic_num = atom.GetAtomicNum()

    atom_type = [1 if atomic_num == n else 0 for n in ATOM_LIST]
    atom_type.append(1 if atomic_num not in ATOM_LIST else 0)

    degree = atom.GetDegree()
    degree_feat = [1 if degree == d else 0 for d in range(6)]

    return np.array(
        atom_type
        + degree_feat
        + [
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs()
        ],
        dtype=np.float32
    )


N_ATOM_FEATURES = 21


def mol_to_graph(mol, max_atoms=MAX_ATOMS):
    n_atoms = mol.GetNumAtoms()

    if n_atoms > max_atoms:
        raise ValueError(f"원자 수가 너무 많습니다. 현재 모델 최대 원자 수: {max_atoms}")

    X = np.zeros((max_atoms, N_ATOM_FEATURES), dtype=np.float32)
    A = np.zeros((max_atoms, max_atoms), dtype=np.float32)

    for i, atom in enumerate(mol.GetAtoms()):
        X[i] = atom_features(atom)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        A[i, j] = 1.0
        A[j, i] = 1.0

    for i in range(n_atoms):
        A[i, i] = 1.0

    degree = A.sum(axis=1)
    degree[degree == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return X.astype(np.float32), A_norm.astype(np.float32)


def predict_logd(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("올바르지 않은 SMILES입니다.")

    X, A = mol_to_graph(mol)

    X = np.expand_dims(X, axis=0)
    A = np.expand_dims(A, axis=0)

    model = load_gnn_model()
    pred = model.predict([X, A], verbose=0).ravel()[0]

    return float(pred)


# =========================
# 구조 변형 후보 생성
# =========================
REACTION_LIBRARY = [
    {
        "name": "Add -OH",
        "group": "-OH",
        "direction": "decrease_logD",
        "rxn": AllChem.ReactionFromSmarts("[cH:1]>>[c:1]O")
    },
    {
        "name": "Add -NH2",
        "group": "-NH₂",
        "direction": "decrease_logD",
        "rxn": AllChem.ReactionFromSmarts("[cH:1]>>[c:1]N")
    },
    {
        "name": "Add -COOH",
        "group": "-COOH",
        "direction": "decrease_logD",
        "rxn": AllChem.ReactionFromSmarts("[cH:1]>>[c:1]C(=O)O")
    },
    {
        "name": "Add -CH3",
        "group": "-CH₃",
        "direction": "increase_logD",
        "rxn": AllChem.ReactionFromSmarts("[cH:1]>>[c:1]C")
    },
    {
        "name": "Add -Cl",
        "group": "-Cl",
        "direction": "increase_logD",
        "rxn": AllChem.ReactionFromSmarts("[cH:1]>>[c:1]Cl")
    },
]


def generate_candidates(smiles, original_logd, max_candidates_per_reaction=10):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return pd.DataFrame()

    if original_logd > TARGET_MAX:
        preferred = "decrease_logD"
    elif original_logd < TARGET_MIN:
        preferred = "increase_logD"
    else:
        preferred = "tune_logD"

    rows = []
    seen = set()

    for info in REACTION_LIBRARY:
        if preferred == "decrease_logD" and info["direction"] != "decrease_logD":
            continue
        if preferred == "increase_logD" and info["direction"] != "increase_logD":
            continue

        try:
            products = info["rxn"].RunReactants((mol,))
        except Exception:
            continue

        count = 0

        for product_set in products:
            if count >= max_candidates_per_reaction:
                break

            product = product_set[0]

            try:
                Chem.SanitizeMol(product)
                cand_smiles = Chem.MolToSmiles(product, canonical=True)
            except Exception:
                continue

            if cand_smiles in seen:
                continue

            seen.add(cand_smiles)

            try:
                cand_logd = predict_logd(cand_smiles)
            except Exception:
                continue

            in_range = TARGET_MIN <= cand_logd <= TARGET_MAX

            if cand_logd < TARGET_MIN:
                penalty = TARGET_MIN - cand_logd
            elif cand_logd > TARGET_MAX:
                penalty = cand_logd - TARGET_MAX
            else:
                penalty = 0

            rows.append({
                "Candidate_SMILES": cand_smiles,
                "Modification": info["name"],
                "Introduced_group": info["group"],
                "Predicted_logD": cand_logd,
                "Delta_logD": cand_logd - original_logd,
                "In_target_range": in_range,
                "Range_penalty": penalty
            })

            count += 1

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    in_range_df = df[df["In_target_range"]].copy()

    if not in_range_df.empty:
        return in_range_df.sort_values(
            ["Range_penalty", "Modification"]
        ).head(6)

    return df.sort_values(
        ["Range_penalty", "Modification"]
    ).head(6)


# =========================
# Streamlit 화면
# =========================
st.title("AI 기반 logD 예측 및 구조 최적화 시스템")

st.markdown(
    """
    SMILES를 입력하면 GNN 모델이 logD를 예측하고,  
    목표 범위(1–3)에 가까워질 수 있는 후보 구조를 추천합니다.
    """
)

smiles = st.text_input("SMILES를 입력하세요", "CC(=O)Oc1ccccc1C(=O)O")

if st.button("logD 예측 및 구조 추천"):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        st.error("올바르지 않은 SMILES입니다.")
    else:
        st.subheader("1. 입력 분자 구조")
        st.write(f"입력 SMILES: `{smiles}`")
        st.image(Draw.MolToImage(mol, size=(420, 300)))

        try:
            original_logd = predict_logd(smiles)

            st.subheader("2. 입력 분자 logD 예측 결과")
            st.metric("GNN 예측 logD", f"{original_logd:.3f}")

            if TARGET_MIN <= original_logd <= TARGET_MAX:
                st.success("입력 분자는 목표 logD 범위(1–3)에 포함됩니다.")
            elif original_logd > TARGET_MAX:
                st.warning("입력 분자의 logD가 목표 범위보다 높습니다. 친수성 작용기 도입 후보를 탐색합니다.")
            else:
                st.warning("입력 분자의 logD가 목표 범위보다 낮습니다. 소수성 작용기 도입 후보를 탐색합니다.")

            st.subheader("3. 추천 후보 구조")

            rec_df = generate_candidates(smiles, original_logd)

            if rec_df.empty:
                st.info("추천 후보가 생성되지 않았습니다. 방향족 C-H 위치가 없는 분자일 수 있습니다.")
            else:
                for i, row in rec_df.iterrows():
                    cand_smiles = row["Candidate_SMILES"]
                    cand_mol = Chem.MolFromSmiles(cand_smiles)

                    with st.container():
                        st.markdown("---")
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.image(Draw.MolToImage(cand_mol, size=(350, 260)))

                        with col2:
                            st.markdown(f"### 후보 구조")
                            st.write(f"도입 작용기: **{row['Introduced_group']}**")
                            st.write(f"변형 방식: {row['Modification']}")
                            st.write(f"후보 SMILES: `{cand_smiles}`")
                            st.metric("후보 예측 logD", f"{row['Predicted_logD']:.3f}")
                            st.write(f"원본 대비 ΔlogD: `{row['Delta_logD']:.3f}`")

                            if row["In_target_range"]:
                                st.success("목표 logD 범위(1–3)에 포함되는 후보입니다.")
                            else:
                                st.info("목표 범위에 가장 가까운 참고 후보입니다.")

        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
