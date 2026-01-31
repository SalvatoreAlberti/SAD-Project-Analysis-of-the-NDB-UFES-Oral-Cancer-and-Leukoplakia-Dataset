# ============================================================
# Logistic Regression (OvR) + TenSEAL (CKKS)
# Training in chiaro + inferenza:
#   - chiaro (pipeline)
#   - chiaro (manuale W,b)
#   - ibrida (solo feature sensibili cifrate)
# Confronto su 5 istanze casuali del test set
# ============================================================

# -------------------------
# IMPORT
# -------------------------
import numpy as np
import pandas as pd
import random
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import tenseal as ts


# -------------------------
# PATH E CONFIGURAZIONE
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_MODIFICATO = PROJECT_ROOT / "dataset" / "dataset_modificato.csv"

# colonne considerate sensibili
SENSITIVE_COLS = ["age_group", "skin_color", "gender"]


# -------------------------
# 1) PREPROCESSOR
# -------------------------
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # colonne numeriche
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # colonne categoriche
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    # one-hot encoder (output denso)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # preprocessing per colonne
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop"
    )

    return preprocessor


# -------------------------
# 2) TRAINING IN CHIARO
# -------------------------
def train_plaintext_model(csv_path: str):
    df = pd.read_csv(csv_path, sep=";")

    # target
    y = df["diagnosis"].astype(str)

    # rimuovi colonne non informative
    drop_cols = [
        "diagnosis", "public_id", "lesion_id", "patient_id", "path",
        "dysplasia_severity", "TaskII", "TaskIII", "TaskIV"
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # split stratificato
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = make_preprocessor(X_train)

    clf = LogisticRegression(
        multi_class="ovr",
        solver="lbfgs",
        max_iter=5000
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", clf)
    ])

    pipe.fit(X_train, y_train)

    # valutazione in chiaro
    y_pred = pipe.predict(X_test)
    print("\n=== VALUTAZIONE IN CHIARO ===")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred))

    # estrazione parametri
    prep_fitted = pipe.named_steps["prep"]
    clf_fitted = pipe.named_steps["clf"]

    classes = clf_fitted.classes_.tolist()
    W = clf_fitted.coef_.astype(np.float64)
    b = clf_fitted.intercept_.astype(np.float64)

    return pipe, prep_fitted, classes, W, b, X_test, y_test


# -------------------------
# 3) CONTESTO TENSEAL CKKS
# -------------------------
def make_tenseal_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context


# -------------------------
# 4) INDICI FEATURE SENSIBILI
# -------------------------
def get_sensitive_indices(prep_fitted, sensitive_cols):
    feature_names = prep_fitted.get_feature_names_out()

    sens_idx = []
    nonsens_idx = []

    for i, name in enumerate(feature_names):
        if any(col in name for col in sensitive_cols):
            sens_idx.append(i)
        else:
            nonsens_idx.append(i)

    return sens_idx, nonsens_idx, feature_names


# -------------------------
# 5A) INFERENZA IN CHIARO (MANUALE)
# -------------------------
def plaintext_predict_manual(prep_fitted, classes, W, b, x_raw_row):
    x_vec = prep_fitted.transform(x_raw_row)
    if hasattr(x_vec, "toarray"):
        x_vec = x_vec.toarray()
    x_vec = x_vec.astype(np.float64)[0]

    scores = np.array([np.dot(x_vec, W[k]) + b[k] for k in range(len(classes))])
    pred_class = classes[int(np.argmax(scores))]

    return pred_class, scores


# -------------------------
# 5B) INFERENZA IBRIDA
# -------------------------
def hybrid_predict(prep_fitted, classes, W, b, x_raw_row,
                   context, sens_idx, nonsens_idx):

    x_full = prep_fitted.transform(x_raw_row)
    if hasattr(x_full, "toarray"):
        x_full = x_full.toarray()
    x_full = x_full.astype(np.float64)[0]

    x_sens = x_full[sens_idx]
    x_non = x_full[nonsens_idx]

    x_sens_enc = ts.ckks_vector(context, x_sens.tolist())

    scores = []
    for k in range(len(classes)):
        w_sens = W[k][sens_idx]
        w_non = W[k][nonsens_idx]

        z_enc = x_sens_enc.dot(w_sens.tolist())
        z_plain = np.dot(x_non, w_non)
        z_final = z_enc + z_plain + b[k]

        scores.append(z_final.decrypt()[0])

    scores = np.array(scores)
    pred_class = classes[int(np.argmax(scores))]

    return pred_class, scores


# -------------------------
# 6) MAIN
# -------------------------
if __name__ == "__main__":

    pipe, prep_fitted, classes, W, b, X_test, y_test = train_plaintext_model(
        str(DATASET_MODIFICATO)
    )

    print("\nClassi:", classes)
    print("W shape:", W.shape, "b shape:", b.shape)

    context = make_tenseal_context()

    sens_idx, nonsens_idx, feature_names = get_sensitive_indices(
        prep_fitted, SENSITIVE_COLS
    )

    print("\nFeature totali:", len(feature_names))
    print("Feature sensibili:", len(sens_idx))
    print("Feature NON sensibili:", len(nonsens_idx))

    # scegli 5 righe casuali
    random_indices = random.sample(range(len(X_test)), 5)

    print("\n======================================")
    print("CONFRONTO SU 5 ISTANZE CASUALI")
    print("======================================")

    for idx in random_indices:
        print("\n--------------------------------------")
        print(f"Istanza test index: {idx}")

        x_one = X_test.iloc[[idx]]
        y_true = y_test.iloc[idx]

        print("Target (vero):", y_true)

        # pipeline sklearn
        pred_pipe = pipe.predict(x_one)[0]
        print("[A] Chiaro (pipeline):", pred_pipe)

        # chiaro manuale
        pred_plain, scores_plain = plaintext_predict_manual(
            prep_fitted, classes, W, b, x_one
        )
        print("[A2] Chiaro (manuale):", pred_plain)

        # ibrido
        pred_hybrid, scores_hybrid = hybrid_predict(
            prep_fitted, classes, W, b, x_one,
            context, sens_idx, nonsens_idx
        )
        print("[B] IBRIDO (sensibili cifrate):", pred_hybrid)

        print("\nScores per classe:")
        for c, sp, sh in zip(classes, scores_plain, scores_hybrid):
            print(f"  {c}")
            print(f"    chiaro : {sp:.6f}")
            print(f"    ibrido : {sh:.6f}")
            print(f"    diff   : {(sh - sp):.2e}")

        print("\nConfronto chiaro vs ibrido:",
              "OK ✅" if pred_pipe == pred_hybrid else "DIVERSO ❌")
