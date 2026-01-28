# ============================================
#  Logistic Regression (OvR) + TenSEAL (CKKS)
#  Training in chiaro + inferenza su dati cifrati
# ============================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import tenseal as ts
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "dataset" / "NDB-UFES An oral cancer and leukoplakia dataset composed of histopathological images and patient data"
DATASET_MODIFICATO = PROJECT_ROOT / "dataset" / "dataset_modificato.csv"

# -----------------------------
# 1) Utility: preprocess robusto
# -----------------------------
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    # OneHotEncoder 
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# -----------------------------
# 2) Training in chiaro (OvR)
# -----------------------------
def train_plaintext_model(csv_path: str):
    df = pd.read_csv(csv_path, sep=";")

    # Target
    y = df["diagnosis"].astype(str)

    # Feature: togliamo ID e path immagine (di solito non sono feature cliniche)
    drop_cols = ["diagnosis", "public_id", "lesion_id", "patient_id", "path","dysplasia_severity","TaskII","TaskIII","TaskIV"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Split stratificato
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = make_preprocessor(X_train)

    # Logistic Regression One-vs-Rest
    clf = LogisticRegression(
        multi_class="ovr",     # 1 classificatore binario per classe
        solver="lbfgs",
        max_iter=5000
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", clf),
    ])

    pipe.fit(X_train, y_train)

    # Valutazione in chiaro (sanity check)
    y_pred = pipe.predict(X_test)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred))

    # Estrazione parametri per inferenza (n_classes x n_features)
    prep_fitted = pipe.named_steps["prep"]
    clf_fitted = pipe.named_steps["clf"]

    classes = clf_fitted.classes_.tolist()
    W = clf_fitted.coef_.astype(np.float64)       # shape: (K, D)
    b = clf_fitted.intercept_.astype(np.float64)  # shape: (K,)

    return pipe, prep_fitted, classes, W, b, X_test


# -----------------------------
# 3) TenSEAL context CKKS
# -----------------------------
def make_tenseal_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context


# -----------------------------
# 4) Inferenza cifrata (OvR)
#    z_k = x·w_k + b_k
# -----------------------------
def encrypted_predict_one(
    context: ts.Context,
    prep_fitted: ColumnTransformer,
    classes: list[str],
    W: np.ndarray,
    b: np.ndarray,
    x_raw_row: pd.DataFrame
):
    """
    x_raw_row: DataFrame con 1 riga, stesse colonne di X (feature grezze, non trasformate)
    """
    # 1) applichi lo stesso preprocessing del training (in chiaro)
    x_vec = prep_fitted.transform(x_raw_row)

    # x_vec può essere sparse: rendiamo denso
    if hasattr(x_vec, "toarray"):
        x_vec = x_vec.toarray()

    x_vec = x_vec.astype(np.float64)[0]  # vettore (D,)

    # 2) cifri SOLO l'input
    x_enc = ts.ckks_vector(context, x_vec.tolist())

    # 3) calcoli gli score cifrati per ogni classe:
    #    z_k = dot(x, w_k) + b_k
    z_enc_list = []
    for k in range(len(classes)):
        wk = W[k].tolist()
        zk_enc = x_enc.dot(wk) + float(b[k])
        z_enc_list.append(zk_enc)

    # 4) decifri gli score (in un caso reale: lo fa il client che ha la secret key)
    z_list = [zk.decrypt()[0] for zk in z_enc_list]  # ogni decrypt -> [val]
    z_arr = np.array(z_list, dtype=np.float64)

    # 5) decisione multiclasse: scegli lo score più alto
    pred_idx = int(np.argmax(z_arr))
    pred_class = classes[pred_idx]

    return pred_class, z_arr


# -----------------------------
# 5) Esecuzione demo
# -----------------------------
if __name__ == "__main__":
    CSV_PATH = DATASET_MODIFICATO 

    # Train in chiaro
    pipe, prep_fitted, classes, W, b, X_test = train_plaintext_model(CSV_PATH)

    print("\nClassi:", classes)
    print("W shape:", W.shape, "b shape:", b.shape)

    # Context TenSEAL
    context = make_tenseal_context()

    # Prendiamo una riga dal test set come esempio di inferenza
    x_one = X_test.iloc[[0]]  # DataFrame 1 riga

    # Inferenza cifrata
    pred_class, scores = encrypted_predict_one(
        context=context,
        prep_fitted=prep_fitted,
        classes=classes,
        W=W,
        b=b,
        x_raw_row=x_one
    )

    print("\n=== Inference cifrata (OvR) ===")
    print("Scores (z_k) per classe:")
    for c, s in zip(classes, scores):
        print(f"  {c}: {s:.6f}")
    print("Predizione finale:", pred_class)

    # Confronto con predizione in chiaro della pipeline (per verifica)
    plain_pred = pipe.predict(x_one)[0]
    print("Predizione in chiaro (pipeline):", plain_pred)
