"""
Evaluacion 2 - Fundamentos de Data Science
Tema: Redes Neuronales y limpieza de datos con fuente CONASET

Este trabajo usa el mismo Excel oficial de CONASET trabajado en la evaluacion
anterior, pero ahora se aplica una red neuronal MLP para predecir siniestros.

Fuente principal:
https://www.conaset.cl/programa/observatorio-datos-estadistica/biblioteca-observatorio/estadisticas-generales/

Fuente secundaria de validacion:
https://mapas-conaset.opendata.arcgis.com
"""
import os
import random
import re
import unicodedata
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# configuracion general
# ---------------------------------------------------------------------
ARCHIVO_EXCEL = "Regionesdeocurrencia2000-2024.xlsx"
API_CONASET = "https://mapas-conaset.opendata.arcgis.com/data.json"
RANDOM_STATE = 42
TARGET = "siniestros"
FEATURES_NUMERICAS = [
    "anio",
    "region_num",
    "fallecidos",
    "lesionados_graves",
    "lesionados_menos_graves",
    "lesionados_leves",
    "total_lesionados",
    "tasa_mortalidad",
    "tasa_lesionados_graves",
    "siniestros_lag_1",
    "siniestros_media_3",
    "total_siniestros_chile_lag_1",
    "periodo_post_pandemia",
]
FEATURES_CATEGORICAS = ["region"]
ULTIMOS_RESULTADOS = None

# ---------------------------------------------------------------------
# validacion de fuente
# ---------------------------------------------------------------------
def verificar_api_conaset():
    """Revisa rapido si la fuente publica de CONASET responde."""
    print("\n[ Validando API publica CONASET / ArcGIS ]")
    try:
        import requests
    except ModuleNotFoundError:
        print("  requests no esta instalado, se continua con el Excel oficial.")
        return False
    try:
        r = requests.get(
            API_CONASET,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=15,
        )
    except Exception as e:
        print(f"  No se pudo conectar con la API: {e}")
        return False
    if r.status_code == 200:
        print("  Conexion exitosa a ArcGIS Open Data (status 200)")
        print("  Fuente publica CONASET verificada")
        return True
    print(f"  La API respondio con status {r.status_code}")
    print("  Se continua con el Excel oficial descargado desde CONASET")
    return False

# ---------------------------------------------------------------------
# carga y limpieza
# ---------------------------------------------------------------------

def normalizar_texto(valor):
    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))

def leer_hoja(df_hoja, anio):
    """Lee una hoja del Excel. El nombre de la hoja se usa como anio."""
    df = df_hoja.iloc[1:].copy().reset_index(drop=True)
    columnas = df_hoja.columns.tolist()
    subheader = df_hoja.iloc[0]
    rename = {columnas[0]: "region"}
    for col in columnas:
        col_norm = normalizar_texto(col)
        if "siniestro" in col_norm:
            rename[col] = "siniestros"
        elif "fallecido" in col_norm:
            rename[col] = "fallecidos"
        elif "total" in col_norm and "lesionado" in col_norm:
            rename[col] = "total_lesionados"

    for i, col in enumerate(columnas):
        if col in rename:
            continue
        sub_norm = normalizar_texto(subheader.iloc[i])
        if sub_norm == "graves":
            rename[col] = "lesionados_graves"
        elif "menos" in sub_norm:
            rename[col] = "lesionados_menos_graves"
        elif sub_norm == "leves":
            rename[col] = "lesionados_leves"
        elif "total" in sub_norm and "lesionado" in sub_norm:
            rename[col] = "total_lesionados"
    anio_limpio = re.sub(r"\D", "", str(anio))
    df = df.rename(columns=rename)
    df["anio"] = int(anio_limpio)
    return df

def cargar_excel():
    if not os.path.exists(ARCHIVO_EXCEL):
        print(f"\nNo se encontro {ARCHIVO_EXCEL}")
        print("Pon el Excel oficial en la misma carpeta que este script.")
        raise SystemExit(1)
    hojas = pd.read_excel(ARCHIVO_EXCEL, sheet_name=None, skiprows=3)
    partes = []
    for nombre_hoja, df_hoja in hojas.items():
        partes.append(leer_hoja(df_hoja, nombre_hoja))

    print(f"\nExcel cargado: {len(hojas)} hojas/anios procesados")
    return pd.concat(partes, ignore_index=True)

def limpiar_datos(df):
    print("\n" + "-" * 55)
    print(" LIMPIEZA DE DATOS")
    print("-" * 55)

    filas_inicio = len(df)

    columnas_numericas = [
        "anio",
        "siniestros",
        "fallecidos",
        "lesionados_graves",
        "lesionados_menos_graves",
        "lesionados_leves",
        "total_lesionados",
    ]

    for col in columnas_numericas:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["region"] = df["region"].astype(str).str.strip()
    df = df.dropna(subset=["anio", "region", "siniestros"])
    df = df[~df["region"].str.contains("total", case=False, na=False)].copy()

    for col in columnas_numericas:
        df[col] = df[col].fillna(0)

    df["anio"] = df["anio"].astype(int)
    df["region_num"] = pd.Categorical(df["region"]).codes + 1

    df["tasa_mortalidad"] = np.where(
        df["siniestros"] > 0,
        (df["fallecidos"] / df["siniestros"] * 100).round(4),
        0,
    )
    df["tasa_lesionados_graves"] = np.where(
        df["siniestros"] > 0,
        (df["lesionados_graves"] / df["siniestros"] * 100).round(4),
        0,
    )

    # Datos del año anterior para no usar informacion del mismo valor a predecir :v
    df = df.sort_values(["region", "anio"]).reset_index(drop=True)
    df["siniestros_lag_1"] = df.groupby("region")["siniestros"].shift(1)
    df["siniestros_media_3"] = (
        df.groupby("region")["siniestros"]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    total_anual = df.groupby("anio")["siniestros"].sum().sort_index()
    total_lag = total_anual.shift(1).rename("total_siniestros_chile_lag_1")
    df = df.merge(total_lag, on="anio", how="left")
    df["periodo_post_pandemia"] = (df["anio"] >= 2021).astype(int)
    df = df.reset_index(drop=True)
    print(f"  filas iniciales: {filas_inicio}")
    print(f"  filas limpias:   {len(df)}")
    print(f"  regiones:        {df['region'].nunique()}")
    print(f"  periodo:         {df['anio'].min()}-{df['anio'].max()}")
    print("-" * 55)
    return df

def preparar_datos():
    df_raw = cargar_excel()
    df_limpio = limpiar_datos(df_raw)
    print("\nDatos limpios cargados en memoria.")
    return df_limpio

# ---------------------------------------------------------------------
# red neuronal
# ---------------------------------------------------------------------
def entrenar_red_neuronal(df):
    global ULTIMOS_RESULTADOS
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        import tensorflow as tf
        from tensorflow import keras
    except ModuleNotFoundError:
        print("\nFaltan librerias para entrenar.")
        print("Ejecuta primero: pip install -r requirements.txt")
        return
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    features = FEATURES_NUMERICAS + FEATURES_CATEGORICAS
    X = df[features].copy()
    y = df[TARGET].astype(float)

    # Chiquillos recuerden que el profe pidio 75-80% para fit y el resto pal test.
    # Lo hacemos temporal: años antiguos para ajustar y recientes para probar.
    anios = sorted(df["anio"].unique())
    n_train = int(round(len(anios) * 0.80))

    anios_train = anios[:n_train]
    anios_test = anios[n_train:]

    mask_train = df["anio"].isin(anios_train)
    mask_test = df["anio"].isin(anios_test)

    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test = X[mask_test], y[mask_test]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURES_NUMERICAS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                FEATURES_CATEGORICAS,
            ),
        ]
    )

    # Primero se separa train/test y despues se ajusta el scaler solo con train :V 
    X_train_prep = preprocesador.fit_transform(X_train)
    X_test_prep = preprocesador.transform(X_test)

    modelo = keras.Sequential(
        [
            keras.layers.Input(shape=(X_train_prep.shape[1],)),
            keras.layers.Dense(64, activation="relu", kernel_initializer="he_uniform"),
            keras.layers.Dropout(0.30),
            keras.layers.Dense(32, activation="relu", kernel_initializer="he_uniform"),
            keras.layers.Dropout(0.30),
            keras.layers.Dense(16, activation="relu", kernel_initializer="he_uniform"),
            keras.layers.Dense(1, activation="relu"),
        ]
    )

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"],
    )

    print("\n" + "=" * 55)
    print(" ENTRENAMIENTO RED NEURONAL")
    print("=" * 55)
    print("Arquitectura: Dense(64) -> Dense(32) -> Dense(16) -> Dense(1)")
    print("Activacion: ReLU")
    print(f"Fit: {len(X_train)} | Test: {len(X_test)}")
    print("Validation split dentro del fit: 15%")
    print(f"Anios fit:   {min(anios_train)}-{max(anios_train)}")
    print(f"Anios test:  {min(anios_test)}-{max(anios_test)}")

    history = modelo.fit(
        X_train_prep,
        y_train,
        validation_split=0.15,
        shuffle=False,
        epochs=150,
        batch_size=16,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=8, factor=0.5, min_lr=1e-5
            ),
        ],
        verbose=1,
    )

    y_pred = modelo.predict(X_test_prep).reshape(-1)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print(" RESULTADOS EN TEST")
    print("=" * 55)
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2  : {r2:.4f}")

    # Tambien pasamos las predicciones a categorias para mostrar otras metricas.
    bins = [0, 2000, 4500, float("inf")]
    labels = ["bajo", "medio", "alto"]
    y_test_cat = pd.cut(y_test, bins=bins, labels=labels)
    y_pred_cat = pd.cut(y_pred, bins=bins, labels=labels)

    accuracy = accuracy_score(y_test_cat, y_pred_cat)
    precision = precision_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0)
    recall = recall_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0)
    f1 = f1_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0)
    matriz = confusion_matrix(y_test_cat, y_pred_cat, labels=labels)
    reporte = classification_report(y_test_cat, y_pred_cat, zero_division=0)

    print("\n" + "=" * 55)
    print(" METRICAS POR CATEGORIA")
    print("=" * 55)
    print("Categorias: bajo < 2000 | medio 2000-4500 | alto > 4500")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nMatriz de confusion:")
    print(pd.DataFrame(matriz, index=labels, columns=labels).to_string())
    print("\nReporte:")
    print(reporte)

    metricas = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "accuracy_categoria": float(accuracy),
        "precision_categoria": float(precision),
        "recall_categoria": float(recall),
        "f1_categoria": float(f1),
        "fit_size": 0.80,
        "validation_split_dentro_fit": 0.15,
        "test_size": 0.20,
        "anios_fit": [int(min(anios_train)), int(max(anios_train))],
        "anios_test": [int(min(anios_test)), int(max(anios_test))],
        "epochs_entrenadas": int(len(history.history["loss"])),
    }

    ULTIMOS_RESULTADOS = {
        "metricas": metricas,
        "historial": pd.DataFrame(history.history),
        "predicciones": pd.DataFrame(
            {
                "y_real": y_test.to_numpy(),
                "y_pred": y_pred,
                "error": y_test.to_numpy() - y_pred,
            }
        ),
        "matriz": pd.DataFrame(matriz, index=labels, columns=labels),
    }

    print("\nResultados listos en memoria.")
    print("Usa la opcion 4 para ver metricas o la opcion 5 para ver graficos.")

# ---------------------------------------------------------------------
# menu simple
# ---------------------------------------------------------------------

def mostrar_metricas_guardadas():
    """Muestra las metricas del ultimo entrenamiento."""
    if ULTIMOS_RESULTADOS is None:
        print("\nTodavia no hay resultados. Primero entrena la red con la opcion 3.")
        return
    metricas = ULTIMOS_RESULTADOS["metricas"]
    print("\n" + "=" * 55)
    print(" METRICAS GUARDADAS")
    print("=" * 55)
    print("[ Regresion ]")
    print(f"MAE : {metricas['mae']:.2f}")
    print(f"MSE : {metricas['mse']:.2f}")
    print(f"RMSE: {metricas['rmse']:.2f}")
    print(f"R2  : {metricas['r2']:.4f}")

    print("\n[ Clasificacion por categorias ]")
    print(f"Accuracy : {metricas['accuracy_categoria']:.4f}")
    print(f"Precision: {metricas['precision_categoria']:.4f}")
    print(f"Recall   : {metricas['recall_categoria']:.4f}")
    print(f"F1-score : {metricas['f1_categoria']:.4f}")

    print("\nMatriz de confusion:")
    print(ULTIMOS_RESULTADOS["matriz"].to_string())

def graficar_resultados():
    """Grafica resultados del ultimo entrenamiento."""
    if ULTIMOS_RESULTADOS is None:
        print("\nTodavia no hay resultados. Primero entrena la red con la opcion 3.")
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("\nFalta matplotlib. Ejecuta: pip install -r requirements.txt")
        return

    historial = ULTIMOS_RESULTADOS["historial"]
    predicciones = ULTIMOS_RESULTADOS["predicciones"]
    matriz = ULTIMOS_RESULTADOS["matriz"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Resultados Red Neuronal CONASET", fontweight="bold")

    axes[0].plot(historial["loss"], label="train")
    axes[0].plot(historial["val_loss"], label="validacion")
    axes[0].set_title("Evolucion de perdida")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    axes[1].scatter(
        predicciones["y_real"],
        predicciones["y_pred"],
        alpha=0.65,
        edgecolors="none",
    )
    minimo = min(predicciones["y_real"].min(), predicciones["y_pred"].min())
    maximo = max(predicciones["y_real"].max(), predicciones["y_pred"].max())
    axes[1].plot([minimo, maximo], [minimo, maximo], "r--", label="prediccion perfecta")
    axes[1].set_title("Real vs predicho")
    axes[1].set_xlabel("Siniestros reales")
    axes[1].set_ylabel("Siniestros predichos")
    axes[1].legend()

    im = axes[2].imshow(matriz.values, cmap="Blues")
    axes[2].set_title("Matriz de confusion")
    axes[2].set_xticks(range(len(matriz.columns)))
    axes[2].set_yticks(range(len(matriz.index)))
    axes[2].set_xticklabels(matriz.columns)
    axes[2].set_yticklabels(matriz.index)
    axes[2].set_xlabel("Predicho")
    axes[2].set_ylabel("Real")

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            axes[2].text(j, i, matriz.iloc[i, j], ha="center", va="center")

    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def main():
    print("\n" + "=" * 60)
    print(" RED NEURONAL CONASET - SINIESTROS VIALES CHILE")
    print("=" * 60)

    verificar_api_conaset()
    df = preparar_datos()

    while True:
        print("\nMENU")
        print("1. Ver ultimas filas")
        print("2. Ver resumen de datos")
        print("3. Entrenar red neuronal")
        print("4. Ver metricas guardadas")
        print("5. Ver graficos de resultados")
        print("6. Salir")

        op = input("\n>> ").strip()

        if op == "1":
            print(df.tail(10).to_string())
        elif op == "2":
            print(df.describe().round(2).to_string())
            print("\nRegiones:", df["region"].nunique())
            print("Anios:", df["anio"].min(), "-", df["anio"].max())
        elif op == "3":
            entrenar_red_neuronal(df)
        elif op == "4":
            mostrar_metricas_guardadas()
        elif op == "5":
            graficar_resultados()
        elif op == "6":
            print("\nListo, fin del programa.")
            break
        else:
            print("Opcion no valida")


if __name__ == "__main__":
    main()
