from pathlib import Path
import os
import numpy as np
import pandas as pd

from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from sklearn.decomposition import PCA


def calcola_feature_pca_da_patch(
    patch_index_path,
    n_components: int = 90,
):
    """
    Legge il file di indice delle patch (patch_index_minimo.csv),
    estrae feature deep da ogni patch con VGG16 pre-addestrata,
    aggrega (media) le feature a livello di immagine (orig_image_path),
    applica PCA e restituisce un DataFrame con:

        orig_image_path, pc1, pc2, ..., pcN

    Parametri
    ---------
    patch_index_path : str o Path
        Path al CSV patch_index_minimo.csv con colonne:
        'patch_path' e 'orig_image_path'.
    n_components : int
        Numero di componenti principali da mantenere (default: 90).

    Ritorna
    -------
    df_pca : pandas.DataFrame
        DataFrame con una riga per immagine:
        colonne: 'orig_image_path', 'pc1', ..., 'pcN'.
    """

    patch_index_path = Path(patch_index_path)

    # ---------- 1. CARICO CSV PATCH ----------
    df_patches = pd.read_csv(patch_index_path)
    

    if "patch_path" not in df_patches.columns or "orig_image_path" not in df_patches.columns:
        raise ValueError("patch_index_minimo.csv deve avere le colonne 'patch_path' e 'orig_image_path'")

    # ---------- 2. CARICO VGG16 PRE-ADDDESTRATA ----------
    print("[INFO] Carico VGG16 pre-addestrata (ImageNet)...")
    base_model = VGG16(weights="imagenet", include_top=False, pooling="avg")
    print("[INFO] Modello caricato.")

    def estrai_feature_patch(patch_path: str) -> np.ndarray:
        """Ritorna un vettore 1D di feature (lunghezza 512) per una patch."""
        img = image.load_img(patch_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = base_model.predict(x, verbose=0)   # (1, 512)
        return feat[0]                            # (512,)

    # ---------- 3. ESTRAGGO FEATURE PER OGNI PATCH ----------
    print("[INFO] Estraggo feature per ogni patch...")

    feature_list = []
    orig_img_list = []

    n_patches = len(df_patches)

    for i, row in df_patches.iterrows():
        patch_path = row["patch_path"]
        orig_image_path = row["orig_image_path"]

        if not os.path.exists(patch_path):
            print(f"[ATTENZIONE] Patch non trovata: {patch_path} (salto)")
            continue

        feat_vec = estrai_feature_patch(patch_path)

        feature_list.append(feat_vec)
        orig_img_list.append(orig_image_path)

        if (i + 1) % 50 == 0:
            print(f"  - Elaborate {i+1}/{n_patches} patch")

    if len(feature_list) == 0:
        raise RuntimeError("Nessuna feature estratta: controlla i path delle patch.")

    feature_array = np.vstack(feature_list)  # (n_patch_effettive, 512)
    print("[INFO] Shape matrice feature delle patch:", feature_array.shape)

    # DataFrame patch-level: 512 feature + orig_image_path
    feature_cols = [f"f{j}" for j in range(feature_array.shape[1])]
    df_feat_patch = pd.DataFrame(feature_array, columns=feature_cols)
    df_feat_patch["orig_image_path"] = orig_img_list

    # ---------- 4. AGGREGO A LIVELLO IMMAGINE ----------
    print("[INFO] Aggrego le feature (media delle patch per immagine)...")

    df_img_feat = (
        df_feat_patch
        .groupby("orig_image_path")[feature_cols]
        .mean()
        .reset_index()
    )

    print("[INFO] Shape feature aggregate per immagine (raw 512):", df_img_feat.shape)

    # ---------- 5. PCA ----------
    if n_components > len(feature_cols):
        raise ValueError(f"n_components={n_components} > numero feature={len(feature_cols)} (max {len(feature_cols)})")

    print(f"[INFO] Applico PCA per ridurre da {len(feature_cols)} a {n_components} componenti...")

    X = df_img_feat[feature_cols].values  # (n_immagini, 512)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)          # (n_immagini, n_components)

    pca_cols = [f"pc{i+1}" for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)

    # aggiungo 'orig_image_path' per poter fare il merge fuori da questa funzione
    df_pca["orig_image_path"] = df_img_feat["orig_image_path"].values

    print("[INFO] Shape feature PCA per immagine:", df_pca.shape)

    return df_pca
