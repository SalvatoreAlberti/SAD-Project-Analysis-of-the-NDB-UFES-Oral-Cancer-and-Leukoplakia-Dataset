import numpy as np
import pandas as pd
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

import tenseal as ts

# >>> aggiunto per grafico
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "dataset" / "dataset_modificato.csv"
SENSITIVE_COLS = ["age_group", "skin_color", "gender"]


def make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "bool"]).columns.tolist()
    

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )


def make_ckks_context() -> ts.Context:
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2**40
    return ctx


def get_sensitive_indices(prep_fitted, sensitive_cols):
    feature_names = prep_fitted.get_feature_names_out()

    sens_idx = [i for i, name in enumerate(feature_names)
                if any(col in name for col in sensitive_cols)]
    nonsens_idx = [i for i in range(len(feature_names)) if i not in set(sens_idx)]

    return sens_idx, nonsens_idx, feature_names


def predict_hybrid(prep_fitted, classes, W, b, x_raw_row: pd.DataFrame,
                   ctx, sens_idx, nonsens_idx):
    # preprocess (chiaro)
    x = prep_fitted.transform(x_raw_row)
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = x.astype(np.float64)[0]

    # split
    x_sens = x[sens_idx]
    x_non = x[nonsens_idx]

    # cifra solo sensibili
    x_sens_enc = ts.ckks_vector(ctx, x_sens.tolist())

    # score per classe
    scores = []
    for k in range(len(classes)):
        w_sens = W[k][sens_idx]
        w_non = W[k][nonsens_idx]

        z_enc = x_sens_enc.dot(w_sens.tolist())     # cifrato
        z_plain = float(np.dot(x_non, w_non))       # chiaro
        z_final = z_enc + z_plain + float(b[k])     # cifrato

        scores.append(z_final.decrypt()[0])         # decifra solo score

    scores = np.array(scores, dtype=np.float64)
    pred = classes[int(np.argmax(scores))]
    return pred, scores


# ==========================================================
# >>> AGGIUNTO: importanza globale delle feature (|pesi|)
# ==========================================================
def plot_global_feature_importance(prep_fitted, W, top_n=15, agg="mean", save_path=None):
    """
    W: coef_ della LogisticRegression, shape (K, D)
    importanza feature = aggregazione su classi di |W|
      - agg="mean": media tra classi
      - agg="max":  massimo tra classi
    """
    feature_names = prep_fitted.get_feature_names_out()
    absW = np.abs(W)  # (K, D)

    if agg == "mean":
        importance = absW.mean(axis=0)  # (D,)
    elif agg == "max":
        importance = absW.max(axis=0)   # (D,)
    else:
        raise ValueError("agg deve essere 'mean' o 'max'")

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    # stampa top 20
    print("\n=== TOP feature (importanza globale = aggregazione |W|) ===")
    print(df_imp.head(20).to_string(index=False))

    # plot top_n
    df_top = df_imp.head(top_n).copy()
    df_top = df_top.iloc[::-1]  # per barh: la più importante in alto

    plt.figure(figsize=(10, 6))
    plt.barh(df_top["feature"], df_top["importance"])
    plt.xlabel(f"Importanza globale (agg={agg} di |peso|)")
    plt.title(f"Top {top_n} feature - Logistic Regression OvR")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"\nGrafico salvato in: {save_path}")

    plt.show()

    return df_imp


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH, sep=";")
    y = df["diagnosis"].astype(str)

    drop_cols = [
        "diagnosis", "public_id", "lesion_id", "patient_id", "path",
        "dysplasia_severity", "TaskII", "TaskIII", "TaskIV"
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    prep = make_preprocessor(X_train)
    X_train_vec = prep.fit_transform(X_train)
    X_test_vec = prep.transform(X_test)

    if hasattr(X_train_vec, "toarray"):
        X_train_vec = X_train_vec.toarray()
    if hasattr(X_test_vec, "toarray"):
        X_test_vec = X_test_vec.toarray()

    X_train_vec = X_train_vec.astype(np.float64)
    X_test_vec = X_test_vec.astype(np.float64)

    clf = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000)
    clf.fit(X_train_vec, y_train)

    classes = clf.classes_.tolist()
    W = clf.coef_.astype(np.float64)
    b = clf.intercept_.astype(np.float64)

    ctx = make_ckks_context()
    sens_idx, nonsens_idx, feature_names = get_sensitive_indices(prep, SENSITIVE_COLS)

    print("Classi:", classes)
    print("Feature totali:", len(feature_names))
    print("Sensibili:", len(sens_idx), "Non sensibili:", len(nonsens_idx))

    # ----------------------------------------------------------
    # >>> AGGIUNTO: importanza globale + grafico
    # ----------------------------------------------------------
    save_fig = PROJECT_ROOT / "feature_importance_lr.png"
    df_imp = plot_global_feature_importance(
        prep_fitted=prep,
        W=W,
        top_n=15,
        agg="mean",            # prova anche "max"
        save_path=save_fig
    )

    # Confronto su 5 istanze
    for idx in random.sample(range(len(X_test)), 5):
        x_one_raw = X_test.iloc[[idx]]
        y_true = y_test.iloc[idx]

        # [A] Predizione sklearn (chiaro) sul vettore già preprocessato
        pred_plain = clf.predict(X_test_vec[idx:idx+1])[0]

        # [B] Predizione ibrida (sensibili cifrate)
        pred_hybrid, scores_h = predict_hybrid(
            prep, classes, W, b, x_one_raw,
            ctx, sens_idx, nonsens_idx
        )

        print("\n--- idx:", idx, "| vero:", y_true, "---")
        print("[A] chiaro (sklearn):", pred_plain)
        print("[B] ibrido (CKKS):  ", pred_hybrid)
        print("chiaro vs ibrido:", "Uguali" if pred_plain == pred_hybrid else "Diversi")
