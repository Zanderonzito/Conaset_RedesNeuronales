"""
TRABAJO DE REDES NEURONALES Y MODELOS PREDICTIVOS DATA SCIENCE
Tema: Seguridad Vial - CONASET

Este trabajo implementa modelos predictivos clasicos (Regresion Lineal,
Regresion Multiple, Random Forest) y una Red Neuronal MLP para predecir
la cantidad de siniestros viales en Chile, utilizando datos oficiales
del Observatorio de CONASET (2000-2024).

Fuente principal:
  https://www.conaset.cl/programa/observatorio-datos-estadistica/
  biblioteca-observatorio/estadisticas-generales/
  Archivo: Regionesdeocurrencia2000-2024.xlsx

Fuente secundaria (validacion):
  https://mapas-conaset.opendata.arcgis.com

Integrantes: Rigo Vega, Martin Caamano, Favio Munoz, Nikolas Maldonado.
"""

import os
import random
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURACION GENERAL
# =====================================================================

# nombre del excel oficial de CONASET, debe estar en la misma carpeta
ARCHIVO_EXCEL = "Regionesdeocurrencia2000-2024.xlsx"

# cache en CSV para no reprocesar el excel cada vez que corremos el script
CSV_CACHE = "datos_conaset.csv"

# url de la API publica de CONASET para validar la fuente de datos
API_CONASET = "https://mapas-conaset.opendata.arcgis.com/data.json"

# semilla fija para que los resultados sean reproducibles
RANDOM_STATE = 42

# variable que queremos predecir
TARGET = "siniestros"

# features para regresion lineal simple: solo el anio
# la idea es ver si existe una tendencia temporal en los siniestros
FEATURES_SIMPLE = ["anio"]

# features para regresion multiple y random forest
# agregamos region, fallecidos y lesionados graves
FEATURES_MULTI = ["anio", "region_num", "fallecidos", "lesionados_graves"]

# features para la red neuronal (necesita mas variables para aprovechar
# la capacidad de aprendizaje de las capas densas)
FEATURES_NUMERICAS_RED = [
    "anio",
    "region_num",
    "fallecidos",
    "lesionados_graves",
    "lesionados_menos_graves",
    "lesionados_leves",
    "total_lesionados",
    "tasa_mortalidad",
    "tasa_lesionados_graves",
    "siniestros_lag_1",         # siniestros del anio anterior
    "siniestros_media_3",      # promedio movil 3 anios anteriores
    "total_siniestros_chile_lag_1",  # total pais del anio anterior
    "periodo_post_pandemia",   # flag binario >= 2021
]
FEATURES_CATEGORICAS_RED = ["region"]

# paleta de colores para todos los graficos (misma en todo el trabajo)
COLORES = {
    "azul":    "#2563EB",
    "rojo":    "#DC2626",
    "verde":   "#16A34A",
    "naranja": "#EA580C",
    "gris":    "#6B7280",
    "morado":  "#7C3AED",
}

# mapeo de regiones a numeros (fijo para que siempre sea el mismo)
REGIONES_NUM = {
    "Tarapacá": 1, "Antofagasta": 2, "Atacama": 3, "Coquimbo": 4,
    "Valparaíso": 5, "L.B.O´Higgins": 6, "Maule": 7, "Biobio": 8,
    "Araucanía": 9, "Los Lagos": 10, "Aysén": 11, "Magallanes": 12,
    "Metropolitana": 13, "Los Ríos": 14, "Arica y Parinacota": 15, "Ñuble": 16,
}

# umbrales para clasificar siniestros en bajo/medio/alto
# bajo: menos de 2000, medio: entre 2000 y 4500, alto: mas de 4500
BINS_CATEGORIA = [0, 2000, 4500, float("inf")]
LABELS_CATEGORIA = ["bajo", "medio", "alto"]

# aqui guardamos los resultados de cada modelo a medida que se entrenan
# al principio estan todos en None porque no se ha entrenado nada
RESULTADOS_CLASICOS = {"simple": None, "multiple": None, "rf": None}
RESULTADOS_RED = None


# =====================================================================
# FUNCIONES DE CARGA Y LIMPIEZA DE DATOS
# =====================================================================

def normalizar_texto(valor):
    """Normaliza texto: convierte a minusculas y elimina tildes."""
    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


def verificar_api_conaset():
    """
    Valida que la fuente publica de CONASET este activa.
    No extraemos datos de aca (el servidor lo bloquea),
    pero sirve para demostrar que la fuente existe y es oficial.
    """
    print("\n[ Validando fuente de datos — API publica CONASET / ArcGIS ]\n")
    try:
        import requests
    except ModuleNotFoundError:
        print("  requests no esta instalado, se continua con el Excel oficial.")
        return
    try:
        r = requests.get(
            API_CONASET,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=15,
        )
        if r.status_code == 200:
            print("  Conexion exitosa a la API de ArcGIS Open Data (status 200)")
            print("  Catalogo de datos CONASET verificado y disponible publicamente")
            print("  Proveedor: CONASET — Ministerio de Transportes y Telecomunicaciones")
        else:
            print(f"  La API respondio pero con codigo {r.status_code}")
            print("  Igual usamos los Excel oficiales que publica el mismo organismo")
    except Exception:
        print("  Sin conexion al servidor de CONASET en este momento")
        print("  Continuamos con los Excel descargados directamente del observatorio")


def leer_hoja(df_hoja, anio):
    """
    Lee una hoja del Excel de CONASET.
    Cada hoja representa un anio y tiene una estructura especifica:
    - fila 0: subheaders (graves, menos graves, leves)
    - columna 0: nombre de la region
    - columnas restantes: siniestros, fallecidos, lesionados por tipo

    El nombre de la hoja (ej: "2024") se usa como anio.
    """
    # la primera fila del df son los subheaders, los datos empiezan en la fila 1
    df = df_hoja.iloc[1:].copy().reset_index(drop=True)
    columnas = df_hoja.columns.tolist()
    subheader = df_hoja.iloc[0]

    # la primera columna siempre es la region
    rename = {columnas[0]: "region"}

    # buscar las columnas principales por el nombre del header
    for col in columnas:
        col_norm = normalizar_texto(col)
        if "siniestro" in col_norm:
            rename[col] = "siniestros"
        elif "fallecido" in col_norm:
            rename[col] = "fallecidos"
        elif "total" in col_norm and "lesionado" in col_norm:
            rename[col] = "total_lesionados"

    # las columnas de lesionados se identifican por el subheader
    # porque el header principal dice "Lesionados" y el subheader dice
    # "graves", "menos graves" o "leves"
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

    # limpiar caracteres no numericos del nombre de la hoja para obtener el anio
    anio_limpio = re.sub(r"\D", "", str(anio))
    df = df.rename(columns=rename)
    df["anio"] = int(anio_limpio)
    return df


def cargar_excel():
    """Lee todas las hojas del Excel oficial de CONASET."""
    if not os.path.exists(ARCHIVO_EXCEL):
        print(f"\n  No se encontro '{ARCHIVO_EXCEL}'")
        print("  Pon el Excel oficial en la misma carpeta que este script.")
        raise SystemExit(1)

    # el Excel tiene una hoja por anio (2000, 2001, ..., 2024)
    # skiprows=3 porque las primeras filas son titulo y encabezados decorativos
    hojas = pd.read_excel(ARCHIVO_EXCEL, sheet_name=None, skiprows=3)
    partes = []
    for nombre_hoja, df_hoja in hojas.items():
        partes.append(leer_hoja(df_hoja, nombre_hoja))

    print(f"\n  Excel cargado: {len(hojas)} hojas/anios procesados")
    return pd.concat(partes, ignore_index=True)


def limpiar_datos(df):
    """
    Limpieza completa del DataFrame:
    1. Convierte columnas a numerico (lo que no sea numero queda como NaN)
    2. Elimina filas con datos faltantes en anio/region/siniestros
    3. Elimina filas de "Total" (son sumas, no observaciones reales)
    4. Codifica la region con el diccionario fijo REGIONES_NUM
    5. Calcula tasas derivadas (mortalidad, lesionados graves)
    6. Crea features temporales avanzadas para la red neuronal
       (lag_1, media_3, total_chile_lag_1, periodo_post_pandemia)
    """
    print("\n" + "-" * 55)
    print("  LIMPIEZA DE DATOS")
    print("-" * 55)
    filas_inicio = len(df)

    # --- paso 1: conversion numerica ---
    # errors='coerce' convierte lo que no sea numero a NaN en vez de dar error
    columnas_numericas = [
        "anio", "siniestros", "fallecidos",
        "lesionados_graves", "lesionados_menos_graves",
        "lesionados_leves", "total_lesionados",
    ]
    for col in columnas_numericas:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- paso 2: eliminar filas invalidas ---
    df["region"] = df["region"].astype(str).str.strip()
    df = df.dropna(subset=["anio", "region", "siniestros"])

    # --- paso 3: eliminar filas de "Total" ---
    # estas filas son agregados (suma de todas las regiones), no observaciones
    # si las dejamos van a inflar los modelos
    df = df[~df["region"].str.contains("total", case=False, na=False)].copy()

    # rellenar nulos restantes con 0 en columnas numericas
    for col in columnas_numericas:
        df[col] = df[col].fillna(0)
    df["anio"] = df["anio"].astype(int)

    # --- paso 4: codificacion de region ---
    # usamos un diccionario fijo para que la codificacion sea siempre la misma
    df["region_num"] = df["region"].map(REGIONES_NUM)
    df["region_num"] = df["region_num"].fillna(-1).astype(int)

    # --- paso 5: features derivadas (tasas) ---
    # tasa_mortalidad: de cada 100 siniestros, cuantos terminaron con muertos
    # np.where para evitar division por cero
    df["tasa_mortalidad"] = np.where(
        df["siniestros"] > 0,
        (df["fallecidos"] / df["siniestros"] * 100).round(4),
        0,
    )
    # tasa_lesionados_graves: de cada 100 siniestros, cuantos con heridos graves
    df["tasa_lesionados_graves"] = np.where(
        df["siniestros"] > 0,
        (df["lesionados_graves"] / df["siniestros"] * 100).round(4),
        0,
    )

    # --- paso 6: features temporales avanzadas ---
    # ordenamos por region y anio para que los shifts funcionen bien
    df = df.sort_values(["region", "anio"]).reset_index(drop=True)

    # siniestros del anio anterior para la misma region
    df["siniestros_lag_1"] = df.groupby("region")["siniestros"].shift(1)

    # promedio movil de los 3 anios anteriores para la misma region
    df["siniestros_media_3"] = (
        df.groupby("region")["siniestros"]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # total de siniestros a nivel pais del anio anterior
    total_anual = df.groupby("anio")["siniestros"].sum().sort_index()
    total_lag = total_anual.shift(1).rename("total_siniestros_chile_lag_1")
    df = df.merge(total_lag, on="anio", how="left")

    # flag binario: 1 si el anio es 2021 o posterior (post-pandemia)
    df["periodo_post_pandemia"] = (df["anio"] >= 2021).astype(int)
    df = df.reset_index(drop=True)

    # --- resumen de limpieza ---
    print(f"  Filas iniciales: {filas_inicio:,}")
    print(f"  Filas limpias:   {len(df):,}")
    print(f"  Regiones:        {df['region'].nunique()}")
    print(f"  Periodo:         {df['anio'].min()}-{df['anio'].max()}")
    print("-" * 55 + "\n")
    return df


def cargar_datos():
    """
    Carga los datos desde el cache CSV si existe,
    o procesa el Excel de CONASET si es la primera vez.
    """
    if os.path.exists(CSV_CACHE):
        df = pd.read_csv(CSV_CACHE)
        print(f"\n[ Datos cargados desde cache — {len(df):,} registros ]\n")
        return df

    print("\n[ Leyendo Excel de CONASET por primera vez... ]\n")
    df = cargar_excel()
    df = limpiar_datos(df)
    df.to_csv(CSV_CACHE, index=False)
    print(f"\n  Datos guardados en '{CSV_CACHE}' para las proximas ejecuciones\n")
    return df


# =====================================================================
# ESTADISTICA DESCRIPTIVA
# =====================================================================

def mostrar_estadistica_descriptiva(df):
    """Muestra promedio, varianza, desviacion estandar, covarianza y correlacion."""
    print("\n" + "=" * 60)
    print("     ESTADISTICA DESCRIPTIVA")
    print("=" * 60)

    cols = ["siniestros", "fallecidos", "lesionados_graves", "tasa_mortalidad"]

    print(f"\n[ Metricas detalladas ]\n")
    print(f"  {'Variable':<20} {'Promedio':>10} {'Varianza':>12} {'Desv.Std':>10}")
    print(f"  {'-' * 56}")
    for col in cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) > 0:
                print(
                    f"  {col:<20} "
                    f"{s.mean():>10.2f} "
                    f"{s.var(ddof=1):>12.2f} "
                    f"{s.std(ddof=1):>10.2f}"
                )

    print(f"\n[ Analisis de varianza ]")
    print(f"  La alta varianza en 'siniestros' ({df['siniestros'].var():.1f}) refleja la disparidad")
    print("  geografica: regiones como la Metropolitana concentran el volumen,")
    print("  mientras zonas extremas muestran valores atipicamente bajos.")

    print(f"\n[ Covarianza y correlacion ]\n")
    pares = [("siniestros", "fallecidos"), ("siniestros", "lesionados_graves")]
    for v1, v2 in pares:
        if v1 in df.columns and v2 in df.columns:
            temp = df[[v1, v2]].copy()
            temp[v1] = pd.to_numeric(temp[v1], errors="coerce")
            temp[v2] = pd.to_numeric(temp[v2], errors="coerce")
            temp = temp.dropna()
            if len(temp) > 1:
                cov = temp.cov().iloc[0, 1]
                corr = temp.corr().iloc[0, 1]
                print(
                    f"  {v1} <-> {v2}: "
                    f"Correlacion = {corr:.4f} "
                    f"(Covarianza = {cov:.2f})"
                )


def hacer_graficos_exploratorios(df):
    """Genera 4 graficos exploratorios en una sola figura."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Analisis Exploratorio — Siniestros Viales Chile (2000-2024)",
        fontsize=14, fontweight="bold",
    )

    # 1. tendencia anual: total de siniestros por anio
    por_anio = df.groupby("anio")["siniestros"].sum().reset_index()
    axes[0, 0].plot(
        por_anio["anio"], por_anio["siniestros"],
        marker="o", color=COLORES["azul"], linewidth=2.5, markersize=6,
    )
    axes[0, 0].fill_between(
        por_anio["anio"], por_anio["siniestros"],
        alpha=0.1, color=COLORES["azul"],
    )
    axes[0, 0].set_title("Evolucion anual de siniestros")
    axes[0, 0].set_xlabel("Año")
    axes[0, 0].set_ylabel("Total siniestros")

    # 2. promedio por region (top 8 regiones con mas siniestros)
    if "region" in df.columns:
        top_reg = (
            df.groupby("region")["siniestros"]
            .mean()
            .sort_values(ascending=True)
            .tail(8)
        )
        colores_barra = [
            COLORES["rojo"] if r == top_reg.idxmax() else COLORES["azul"]
            for r in top_reg.index
        ]
        axes[0, 1].barh(
            top_reg.index, top_reg.values,
            color=colores_barra, edgecolor="white",
        )
        axes[0, 1].set_title("Promedio de siniestros por region (top 8)")
        axes[0, 1].set_xlabel("Promedio anual")

    # 3. mapa de calor de correlaciones entre variables principales
    cols_corr = [
        c for c in ["siniestros", "fallecidos", "lesionados_graves",
                     "lesionados_leves", "tasa_mortalidad"]
        if c in df.columns
    ]
    sns.heatmap(
        df[cols_corr].corr(), annot=True, fmt=".2f",
        cmap="Blues", ax=axes[1, 0], linewidths=0.5, annot_kws={"size": 9},
    )
    axes[1, 0].set_title("Mapa de calor — correlaciones")

    # 4. scatter de siniestros vs fallecidos con linea de tendencia
    muestra = df.sample(min(600, len(df)), random_state=42)
    axes[1, 1].scatter(
        muestra["siniestros"], muestra["fallecidos"],
        alpha=0.4, color=COLORES["verde"], edgecolors="none", s=20,
    )
    m, b = np.polyfit(df["siniestros"], df["fallecidos"], 1)
    xs = np.linspace(df["siniestros"].min(), df["siniestros"].max(), 100)
    corr_val = df[["siniestros", "fallecidos"]].corr().iloc[0, 1]
    axes[1, 1].plot(
        xs, m * xs + b, color=COLORES["rojo"],
        linewidth=2, label=f"tendencia (r={corr_val:.2f})",
    )
    axes[1, 1].set_title("Siniestros vs Fallecidos")
    axes[1, 1].set_xlabel("Siniestros")
    axes[1, 1].set_ylabel("Fallecidos")
    axes[1, 1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("graficos_exploratorios.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Grafico guardado como 'graficos_exploratorios.png'")


# =====================================================================
# FUNCIONES COMUNES PARA MODELOS
# =====================================================================

def preparar_split_aleatorio(df, features):
    """
    Prepara los datos y hace split 80/20 aleatorio para modelos clasicos.
    - Elimina filas con nulos en las features o en el target
    - Separa X (features) e y (target)
    - Divide en train (80%) y test (20%) con semilla fija
    """
    df_limpio = df.dropna(subset=features + [TARGET]).copy()
    X = df_limpio[features]
    y = df_limpio[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def imprimir_metricas(nombre, y_train, y_pred_train, y_test, y_pred_test):
    """
    Imprime R2, RMSE y MAE para train y test de cualquier modelo.
    Tambien detecta posible sobreajuste comparando R2 de train vs test.
    Retorna un diccionario con todas las metricas para la comparacion final.
    """
    r2_tr = r2_score(y_train, y_pred_train)
    r2_te = r2_score(y_test, y_pred_test)
    rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_tr = mean_absolute_error(y_train, y_pred_train)
    mae_te = mean_absolute_error(y_test, y_pred_test)
    gap = r2_tr - r2_te

    print(f"\n  Metricas — {nombre}")
    print(f"  {'':10} {'train':>10} {'test':>10}")
    print(f"  {'R2':10} {r2_tr:>10.4f} {r2_te:>10.4f}")
    print(f"  {'RMSE':10} {rmse_tr:>10.2f} {rmse_te:>10.2f}")
    print(f"  {'MAE':10} {mae_tr:>10.2f} {mae_te:>10.2f}")

    # si la diferencia entre R2 de train y test es mayor a 0.15,
    # puede haber sobreajuste (el modelo memorizo los datos de train)
    if abs(gap) > 0.15:
        print(f"  aviso: gap R2 = {gap:+.4f} — puede haber sobreajuste")
    else:
        print(f"  gap R2 = {gap:+.4f} — ok, generaliza bien")

    return {
        "r2_train": r2_tr, "r2_test": r2_te,
        "rmse_train": rmse_tr, "rmse_test": rmse_te,
        "mae_train": mae_tr, "mae_test": mae_te,
    }


def graficar_diagnostico(y_test, y_pred, nombre):
    """
    Grafico de diagnostico con 2 paneles:
    - Izquierda: valores reales vs predichos (lo ideal es que queden sobre la linea roja)
    - Derecha: histograma de residuos (lo ideal es que esten centrados en 0)
    """
    residuos = y_test.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Diagnostico — {nombre}", fontweight="bold")

    # panel 1: real vs predicho
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    axes[0].scatter(
        y_test, y_pred, alpha=0.4, color=COLORES["azul"],
        edgecolors="none", s=25,
    )
    axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="prediccion perfecta")
    axes[0].set_xlabel("Valor real")
    axes[0].set_ylabel("Valor predicho")
    axes[0].set_title("Real vs Predicho")
    axes[0].legend(fontsize=9)

    # panel 2: histograma de residuos
    axes[1].hist(
        residuos, bins=25, color=COLORES["azul"],
        edgecolor="white", alpha=0.8,
    )
    axes[1].axvline(0, color=COLORES["rojo"], linestyle="--", linewidth=1.5,
                    label="residuo = 0")
    axes[1].axvline(residuos.mean(), color=COLORES["naranja"], linewidth=1.5,
                    label=f"media = {residuos.mean():.2f}")
    axes[1].set_xlabel("Residuo")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Histograma de residuos")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()


def matriz_confusion_categorias(y_test, y_pred, nombre):
    """
    Convierte las predicciones de regresion a categorias (bajo/medio/alto)
    y muestra la matriz de confusion.
    Esto sirve para ver si el modelo al menos acerto la categoria general,
    aunque no haya acertado el numero exacto.
    """
    y_test_s = pd.Series(np.array(y_test).flatten()).reset_index(drop=True)
    y_pred_s = pd.Series(np.array(y_pred).flatten()).reset_index(drop=True)

    y_test_cat = pd.cut(y_test_s, bins=BINS_CATEGORIA, labels=LABELS_CATEGORIA)
    y_pred_cat = pd.cut(y_pred_s, bins=BINS_CATEGORIA, labels=LABELS_CATEGORIA)

    mask = y_test_cat.notna() & y_pred_cat.notna()
    y_test_cat = y_test_cat[mask].astype(str).values
    y_pred_cat = y_pred_cat[mask].astype(str).values

    cm = confusion_matrix(y_test_cat, y_pred_cat, labels=LABELS_CATEGORIA)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS_CATEGORIA)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de Confusion — {nombre}")
    plt.tight_layout()
    plt.show()


# =====================================================================
# MODELOS CLASICOS
# =====================================================================

def _entrenar_simple(df):
    """
    Modelo 1: Regresion Lineal Simple
    Solo usa el anio como variable predictora.
    Sirve para ver si hay una tendencia temporal en los siniestros.
    Ecuacion: siniestros = coeficiente * anio + intercepto
    """
    print("\n" + "=" * 55)
    print("  MODELO 1 — Regresion Lineal Simple (anio -> siniestros)")
    print("=" * 55)

    # separar datos en 80% entrenamiento y 20% prueba
    X_tr, X_te, y_tr, y_te = preparar_split_aleatorio(df, FEATURES_SIMPLE)

    # crear y entrenar el modelo
    mod = LinearRegression()
    mod.fit(X_tr, y_tr)

    # mostrar la ecuacion que aprendio el modelo
    coef = mod.coef_[0]
    print(f"\n  Ecuacion: siniestros = {coef:.2f} * anio + ({mod.intercept_:.2f})")
    print(f"  Por cada anio que pasa, los siniestros cambian en {coef:.2f} unidades")

    # hacer predicciones y calcular metricas
    y_pred_train = mod.predict(X_tr)
    y_pred_test = mod.predict(X_te)
    metricas = imprimir_metricas("Lineal Simple", y_tr, y_pred_train, y_te, y_pred_test)

    # validacion cruzada 5-fold para verificar que el modelo es estable
    scores = cross_val_score(mod, X_tr, y_tr, cv=5, scoring="r2")
    print(f"  Cross Validation R2: {scores.mean():.4f} +/- {scores.std():.4f}")

    # graficos de diagnostico
    graficar_diagnostico(y_te, y_pred_test, "Regresion Lineal Simple")
    matriz_confusion_categorias(y_te, y_pred_test, "Regresion Lineal Simple")

    # guardar resultados para la comparacion final
    RESULTADOS_CLASICOS["simple"] = (mod, metricas, FEATURES_SIMPLE)
    return mod, metricas


def _entrenar_multiple(df):
    """
    Modelo 2: Regresion Lineal Multiple
    Usa anio, region, fallecidos y lesionados graves como predictores.
    Los coeficientes indican cuanto aporta cada variable al resultado.
    """
    feats_ok = [f for f in FEATURES_MULTI if f in df.columns]
    print("\n" + "=" * 55)
    print(f"  MODELO 2 — Regresion Multiple")
    print(f"  Features: {feats_ok}")
    print("=" * 55)

    # separar datos en 80% entrenamiento y 20% prueba
    X_tr, X_te, y_tr, y_te = preparar_split_aleatorio(df, feats_ok)

    # crear y entrenar el modelo
    mod = LinearRegression()
    mod.fit(X_tr, y_tr)

    # mostrar los coeficientes: cuanto aporta cada variable
    print("\n  Coeficientes (cuanto aporta cada variable):")
    for feat, c in zip(feats_ok, mod.coef_):
        print(f"    {feat:<22}: {c:+.4f}")

    # hacer predicciones y calcular metricas
    y_pred_train = mod.predict(X_tr)
    y_pred_test = mod.predict(X_te)
    metricas = imprimir_metricas("Regresion Multiple", y_tr, y_pred_train, y_te, y_pred_test)

    # validacion cruzada 5-fold
    scores = cross_val_score(mod, X_tr, y_tr, cv=5, scoring="r2")
    print(f"  Cross Validation R2: {scores.mean():.4f} +/- {scores.std():.4f}")

    # graficos de diagnostico
    graficar_diagnostico(y_te, y_pred_test, "Regresion Multiple")
    matriz_confusion_categorias(y_te, y_pred_test, "Regresion Multiple")

    # guardar resultados
    RESULTADOS_CLASICOS["multiple"] = (mod, metricas, feats_ok)
    return mod, metricas


def _entrenar_rf(df):
    """
    Modelo 3: Random Forest (100 arboles, profundidad max 3)
    Captura relaciones no lineales entre variables.
    La importancia de variables nos dice cuales features son mas relevantes.
    """
    feats_ok = [f for f in FEATURES_MULTI if f in df.columns]
    print("\n" + "=" * 55)
    print("  MODELO 3 — Random Forest (100 arboles, profundidad max 3)")
    print("=" * 55)

    # separar datos en 80% entrenamiento y 20% prueba
    X_tr, X_te, y_tr, y_te = preparar_split_aleatorio(df, feats_ok)
    print("  Entrenando... puede tardar unos segundos")

    # crear y entrenar el modelo con hiperparametros conservadores
    # para evitar sobreajuste en un dataset pequeño (~400 filas)
    mod = RandomForestRegressor(
        n_estimators=100,    # 100 arboles en el bosque
        max_depth=3,         # profundidad maxima de cada arbol
        min_samples_split=10,  # minimo 10 muestras para dividir un nodo
        min_samples_leaf=5,    # minimo 5 muestras en cada hoja
        random_state=RANDOM_STATE,
        n_jobs=-1,  # usar todos los nucleos del procesador
    )
    mod.fit(X_tr, y_tr)

    # mostrar la importancia de cada variable
    print("\n  Importancia de variables:")
    importancias = pd.Series(mod.feature_importances_, index=feats_ok)
    for feat, imp in importancias.sort_values(ascending=False).items():
        barra = "█" * int(imp * 35)
        print(f"    {feat:<22}: {barra} {imp:.4f}")

    # hacer predicciones y calcular metricas
    y_pred_train = mod.predict(X_tr)
    y_pred_test = mod.predict(X_te)
    metricas = imprimir_metricas("Random Forest", y_tr, y_pred_train, y_te, y_pred_test)

    # validacion cruzada 5-fold
    scores = cross_val_score(mod, X_tr, y_tr, cv=5, scoring="r2")
    print(f"  Cross Validation R2: {scores.mean():.4f} +/- {scores.std():.4f}")

    # graficos de diagnostico
    graficar_diagnostico(y_te, y_pred_test, "Random Forest")
    matriz_confusion_categorias(y_te, y_pred_test, "Random Forest")

    # guardar resultados
    RESULTADOS_CLASICOS["rf"] = (mod, metricas, feats_ok)
    return mod, metricas


# =====================================================================
# RED NEURONAL MLP
# =====================================================================

def entrenar_red_neuronal(df):
    """
    Red neuronal MLP (Multi-Layer Perceptron) para predecir siniestros.

    Arquitectura:
      Input -> Dense(64, ReLU) -> Dropout(0.3)
            -> Dense(32, ReLU) -> Dropout(0.3)
            -> Dense(16, ReLU)
            -> Dense(1, ReLU)

    Se usa ReLU en la salida porque los siniestros no pueden ser negativos.

    Split temporal:
      - 75-80% de los anios mas antiguos para fit (entrenamiento)
      - 20-25% de los anios mas recientes para test
      - Dentro del fit se reserva 15% para validacion interna

    El scaler se ajusta SOLO con los datos de fit para evitar data leakage
    (no queremos que el modelo "vea" datos del futuro durante el entrenamiento).
    """
    global RESULTADOS_RED

    # importar tensorflow dentro de la funcion para que el resto del script
    # funcione aunque tensorflow no este instalado
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        import tensorflow as tf
        from tensorflow import keras
    except ModuleNotFoundError:
        print("\n  Faltan librerias para entrenar la red neuronal.")
        print("  Ejecuta primero: pip install tensorflow")
        return

    # fijar semillas para reproducibilidad
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # preparar features y target
    features = FEATURES_NUMERICAS_RED + FEATURES_CATEGORICAS_RED
    X = df[features].copy()
    y = df[TARGET].astype(float)

    # split temporal: anios antiguos para fit, recientes para test
    # usamos 80% de los anios para fit y 20% para test
    anios = sorted(df["anio"].unique())
    n_train = int(round(len(anios) * 0.80))

    anios_train = anios[:n_train]
    anios_test = anios[n_train:]

    mask_train = df["anio"].isin(anios_train)
    mask_test = df["anio"].isin(anios_test)

    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test = X[mask_test], y[mask_test]

    # preprocesamiento con ColumnTransformer:
    # - numericas: imputar nulos con mediana + escalar con StandardScaler
    # - categoricas: imputar nulos con valor mas frecuente + OneHotEncoder
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                FEATURES_NUMERICAS_RED,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", encoder),
                ]),
                FEATURES_CATEGORICAS_RED,
            ),
        ]
    )

    # fit_transform solo con datos de entrenamiento (evitar data leakage)
    # transform con datos de test usando los parametros aprendidos del train
    X_train_prep = preprocesador.fit_transform(X_train)
    X_test_prep = preprocesador.transform(X_test)

    # construir la red neuronal con Keras Sequential API
    # capas Dense con activacion ReLU y Dropout para regularizacion
    # kernel_initializer='he_uniform' es recomendado para redes con ReLU
    modelo = keras.Sequential([
        keras.layers.Input(shape=(X_train_prep.shape[1],)),
        keras.layers.Dense(64, activation="relu", kernel_initializer="he_uniform"),
        keras.layers.Dropout(0.30),
        keras.layers.Dense(32, activation="relu", kernel_initializer="he_uniform"),
        keras.layers.Dropout(0.30),
        keras.layers.Dense(16, activation="relu", kernel_initializer="he_uniform"),
        keras.layers.Dense(1, activation="relu"),
    ])

    # compilar con Adam optimizer y MSE como funcion de perdida
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"],
    )

    # mostrar configuracion del entrenamiento
    print("\n" + "=" * 55)
    print("  ENTRENAMIENTO RED NEURONAL")
    print("=" * 55)
    print("  Arquitectura: Dense(64) -> Dense(32) -> Dense(16) -> Dense(1)")
    print("  Activacion: ReLU | Dropout: 0.30")
    print(f"  Fit: {len(X_train)} observaciones | Test: {len(X_test)} observaciones")
    print(f"  Proporcion fit/test: {len(X_train)/(len(X_train)+len(X_test))*100:.0f}% / {len(X_test)/(len(X_train)+len(X_test))*100:.0f}%")
    print("  Validation split dentro del fit: 15%")
    print(f"  Anios fit:  {min(anios_train)}-{max(anios_train)}")
    print(f"  Anios test: {min(anios_test)}-{max(anios_test)}")

    # entrenar la red neuronal
    # EarlyStopping: si la val_loss no mejora en 20 epocas, para y restaura los mejores pesos
    # ReduceLROnPlateau: si la val_loss se estanca 8 epocas, reduce el learning rate a la mitad
    history = modelo.fit(
        X_train_prep,
        y_train,
        validation_split=0.15,  # 15% del fit se usa para validacion interna
        shuffle=False,  # no mezclar porque es temporal
        epochs=150,
        batch_size=16,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=8, factor=0.5, min_lr=1e-5,
            ),
        ],
        verbose=1,
    )

    # hacer predicciones en test
    y_pred = modelo.predict(X_test_prep).reshape(-1)

    # calcular metricas de regresion
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # metricas de train para detectar sobreajuste
    y_pred_train = modelo.predict(X_train_prep).reshape(-1)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

    print("\n" + "=" * 55)
    print("  RESULTADOS EN TEST")
    print("=" * 55)
    print(f"  MAE :  {mae:.2f}")
    print(f"  MSE :  {mse:.2f}")
    print(f"  RMSE:  {rmse:.2f}")
    print(f"  R2  :  {r2:.4f}")

    gap = r2_train - r2
    if abs(gap) > 0.15:
        print(f"  aviso: gap R2 = {gap:+.4f} — puede haber sobreajuste")
    else:
        print(f"  gap R2 = {gap:+.4f} — ok, generaliza bien")

    # convertir predicciones a categorias bajo/medio/alto
    # para calcular metricas de clasificacion adicionales
    y_test_cat = pd.cut(y_test, bins=BINS_CATEGORIA, labels=LABELS_CATEGORIA)
    y_pred_cat = pd.cut(y_pred, bins=BINS_CATEGORIA, labels=LABELS_CATEGORIA)

    accuracy = accuracy_score(y_test_cat, y_pred_cat)
    precision = precision_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0)
    recall = recall_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0)
    f1 = f1_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0)
    matriz = confusion_matrix(y_test_cat, y_pred_cat, labels=LABELS_CATEGORIA)
    reporte = classification_report(y_test_cat, y_pred_cat, zero_division=0)

    print("\n" + "=" * 55)
    print("  METRICAS POR CATEGORIA")
    print("=" * 55)
    print("  Categorias: bajo < 2000 | medio 2000-4500 | alto > 4500")
    print(f"  Accuracy :  {accuracy:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall   :  {recall:.4f}")
    print(f"  F1-score :  {f1:.4f}")
    print("\n  Matriz de confusion:")
    print(pd.DataFrame(matriz, index=LABELS_CATEGORIA, columns=LABELS_CATEGORIA).to_string())
    print("\n  Reporte de clasificacion:")
    print(reporte)

    # guardar todo en memoria para consultas posteriores
    RESULTADOS_RED = {
        "metricas": {
            "r2_test": float(r2), "r2_train": float(r2_train),
            "rmse_test": float(rmse), "rmse_train": float(rmse_train),
            "mae_test": float(mae), "mse_test": float(mse),
            "accuracy_categoria": float(accuracy),
            "precision_categoria": float(precision),
            "recall_categoria": float(recall),
            "f1_categoria": float(f1),
            "anios_fit": [int(min(anios_train)), int(max(anios_train))],
            "anios_test": [int(min(anios_test)), int(max(anios_test))],
            "epochs_entrenadas": int(len(history.history["loss"])),
        },
        "historial": pd.DataFrame(history.history),
        "predicciones": pd.DataFrame({
            "y_real": y_test.to_numpy(),
            "y_pred": y_pred,
            "error": y_test.to_numpy() - y_pred,
        }),
        "matriz": pd.DataFrame(matriz, index=LABELS_CATEGORIA, columns=LABELS_CATEGORIA),
    }

    print("\n  Resultados guardados en memoria.")
    print("  Usa opcion 9 para metricas o 10 para graficos.")


def mostrar_metricas_red():
    """Muestra las metricas del ultimo entrenamiento de la red neuronal."""
    if RESULTADOS_RED is None:
        print("\n  Todavia no hay resultados. Primero entrena la red con la opcion 8.")
        return

    m = RESULTADOS_RED["metricas"]
    print("\n" + "=" * 55)
    print("  METRICAS RED NEURONAL (GUARDADAS)")
    print("=" * 55)
    print("  [ Regresion ]")
    print(f"  MAE :  {m['mae_test']:.2f}")
    print(f"  RMSE:  {m['rmse_test']:.2f}")
    print(f"  R2  :  {m['r2_test']:.4f}")
    print(f"\n  [ Clasificacion por categorias ]")
    print(f"  Accuracy :  {m['accuracy_categoria']:.4f}")
    print(f"  Precision:  {m['precision_categoria']:.4f}")
    print(f"  Recall   :  {m['recall_categoria']:.4f}")
    print(f"  F1-score :  {m['f1_categoria']:.4f}")
    print(f"\n  Matriz de confusion:")
    print(RESULTADOS_RED["matriz"].to_string())


def graficar_resultados_red():
    """
    Grafica 3 paneles del entrenamiento de la red neuronal:
    1. Evolucion de la perdida (train vs validacion)
    2. Real vs predicho
    3. Matriz de confusion
    """
    if RESULTADOS_RED is None:
        print("\n  Todavia no hay resultados. Primero entrena la red con la opcion 8.")
        return

    historial = RESULTADOS_RED["historial"]
    predicciones = RESULTADOS_RED["predicciones"]
    matriz = RESULTADOS_RED["matriz"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Resultados Red Neuronal — CONASET", fontweight="bold")

    # 1. evolucion de la perdida por epoca
    axes[0].plot(historial["loss"], label="train", color=COLORES["azul"])
    axes[0].plot(historial["val_loss"], label="validacion", color=COLORES["naranja"])
    axes[0].set_title("Evolucion de perdida")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    # 2. scatter real vs predicho
    axes[1].scatter(
        predicciones["y_real"], predicciones["y_pred"],
        alpha=0.65, edgecolors="none", color=COLORES["azul"],
    )
    minimo = min(predicciones["y_real"].min(), predicciones["y_pred"].min())
    maximo = max(predicciones["y_real"].max(), predicciones["y_pred"].max())
    axes[1].plot([minimo, maximo], [minimo, maximo], "r--", label="prediccion perfecta")
    axes[1].set_title("Real vs predicho")
    axes[1].set_xlabel("Siniestros reales")
    axes[1].set_ylabel("Siniestros predichos")
    axes[1].legend()

    # 3. matriz de confusion por categorias
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
    plt.savefig("resultados_red_neuronal.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Grafico guardado como 'resultados_red_neuronal.png'")


# =====================================================================
# COMPARACION DE TODOS LOS MODELOS
# =====================================================================

def comparar_todos_los_modelos():
    """
    Compara los 4 modelos (3 clasicos + red neuronal) en una tabla con:
    - R2 test, RMSE test, MAE test
    - Interpretabilidad, tiempo de entrenamiento, riesgo de sobreajuste
    - Grafico de barras comparativo de R2, RMSE y MAE
    - Reporte de texto con recomendaciones

    Prerequisito: los 4 modelos deben estar entrenados.
    """
    # verificar que los modelos clasicos esten entrenados
    faltantes_clasicos = [k for k, v in RESULTADOS_CLASICOS.items() if v is None]
    if faltantes_clasicos:
        nombres_menu = {"simple": "4", "multiple": "5", "rf": "6"}
        print("\n  Faltan modelos clasicos por entrenar:")
        for f in faltantes_clasicos:
            print(f"    -> opcion {nombres_menu[f]} ({f})")
        return

    # verificar que la red neuronal este entrenada
    if RESULTADOS_RED is None:
        print("\n  Falta entrenar la red neuronal.")
        print("    -> opcion 8")
        return

    print("\n" + "=" * 70)
    print("  COMPARATIVA DE TODOS LOS MODELOS")
    print("=" * 70)

    # recopilar metricas de cada modelo
    modelos = {
        "Lineal Simple": RESULTADOS_CLASICOS["simple"][1],
        "Reg. Multiple": RESULTADOS_CLASICOS["multiple"][1],
        "Random Forest": RESULTADOS_CLASICOS["rf"][1],
        "Red Neuronal":  {
            "r2_test": RESULTADOS_RED["metricas"]["r2_test"],
            "rmse_test": RESULTADOS_RED["metricas"]["rmse_test"],
            "mae_test": RESULTADOS_RED["metricas"]["mae_test"],
        },
    }

    # caracteristicas cualitativas de cada modelo
    interpretabilidad = {
        "Lineal Simple": "Muy alta",
        "Reg. Multiple": "Alta",
        "Random Forest": "Media",
        "Red Neuronal":  "Baja",
    }
    tiempo_entreno = {
        "Lineal Simple": "<1s",
        "Reg. Multiple": "<1s",
        "Random Forest": "~5s",
        "Red Neuronal":  "~30s",
    }
    riesgo_overfit = {
        "Lineal Simple": "Bajo",
        "Reg. Multiple": "Bajo",
        "Random Forest": "Medio",
        "Red Neuronal":  "Posible",
    }

    # imprimir tabla comparativa en consola
    header = (
        f"  {'Modelo':<18} {'R2 Test':>10} {'RMSE Test':>12} {'MAE Test':>10} "
        f"{'Interpret.':>14} {'Tiempo':>8} {'Overfit':>10}"
    )
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")

    for nombre, m in modelos.items():
        print(
            f"  {nombre:<18} {m['r2_test']:>10.4f} {m['rmse_test']:>12.2f} "
            f"{m['mae_test']:>10.2f} {interpretabilidad[nombre]:>14} "
            f"{tiempo_entreno[nombre]:>8} {riesgo_overfit[nombre]:>10}"
        )

    # ranking por R2 (de mayor a menor)
    ranking = sorted(modelos.items(), key=lambda x: x[1]["r2_test"], reverse=True)
    print(f"\n  Mejor modelo por R2: {ranking[0][0]} ({ranking[0][1]['r2_test']:.4f})")

    # grafico comparativo con 3 paneles: R2, RMSE, MAE
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Comparativa de Modelos — Siniestralidad Vial Chile",
        fontsize=13, fontweight="bold",
    )

    nombres_graf = list(modelos.keys())
    colores = [COLORES["naranja"], COLORES["azul"], COLORES["verde"], COLORES["morado"]]

    # panel 1: R2 por modelo (mas alto = mejor)
    r2_vals = [m["r2_test"] for m in modelos.values()]
    bars = axes[0].bar(nombres_graf, r2_vals, color=colores, edgecolor="white", width=0.5)
    for bar, val in zip(bars, r2_vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )
    axes[0].set_ylabel("R2 Test")
    axes[0].set_title("R2 por modelo")
    axes[0].set_ylim(0, max(r2_vals) * 1.3 if max(r2_vals) > 0 else 1)
    axes[0].tick_params(axis="x", rotation=15)

    # panel 2: RMSE por modelo (mas bajo = mejor)
    rmse_vals = [m["rmse_test"] for m in modelos.values()]
    bars2 = axes[1].bar(nombres_graf, rmse_vals, color=colores, edgecolor="white", width=0.5)
    for bar, val in zip(bars2, rmse_vals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rmse_vals) * 0.01,
            f"{val:.2f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )
    axes[1].set_ylabel("RMSE Test")
    axes[1].set_title("RMSE por modelo (menor es mejor)")
    axes[1].tick_params(axis="x", rotation=15)

    # panel 3: MAE por modelo (mas bajo = mejor)
    mae_vals = [m["mae_test"] for m in modelos.values()]
    bars3 = axes[2].bar(nombres_graf, mae_vals, color=colores, edgecolor="white", width=0.5)
    for bar, val in zip(bars3, mae_vals):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mae_vals) * 0.01,
            f"{val:.2f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )
    axes[2].set_ylabel("MAE Test")
    axes[2].set_title("MAE por modelo (menor es mejor)")
    axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig("comparativa_modelos.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Grafico guardado como 'comparativa_modelos.png'")

    # generar reporte de texto con analisis y recomendaciones
    reporte = []
    reporte.append("=" * 70)
    reporte.append("REPORTE — COMPARATIVA DE TODOS LOS MODELOS")
    reporte.append("Trabajo de Redes Neuronales y Modelos Predictivos Data Science")
    reporte.append("Datos: CONASET — Siniestralidad Vial Chile (2000-2024)")
    reporte.append("=" * 70)
    reporte.append("")
    reporte.append("TABLA DE RESULTADOS")
    reporte.append(f"{'Modelo':<18} {'R2 Test':>10} {'RMSE Test':>12} {'MAE Test':>10}")
    reporte.append("-" * 52)
    for nombre, m in modelos.items():
        reporte.append(
            f"{nombre:<18} {m['r2_test']:>10.4f} {m['rmse_test']:>12.2f} {m['mae_test']:>10.2f}"
        )
    reporte.append("")
    reporte.append(f"Mejor modelo por R2: {ranking[0][0]} ({ranking[0][1]['r2_test']:.4f})")
    reporte.append("")
    reporte.append("ANALISIS POR MODELO")
    reporte.append("")
    reporte.append("Regresion Lineal Simple:")
    reporte.append("  Ideal para explicar tendencias globales en el tiempo.")
    reporte.append("  Requiere pocos datos y es facil de interpretar.")
    reporte.append("  Limitado: no captura relaciones no lineales entre variables.")
    reporte.append("")
    reporte.append("Regresion Multiple:")
    reporte.append("  Permite cuantificar el efecto de cada variable con coeficientes.")
    reporte.append("  Los coeficientes tienen interpretacion directa y real.")
    reporte.append("  Recomendado para reportes publicos de CONASET.")
    reporte.append("")
    reporte.append("Random Forest:")
    reporte.append("  Captura interacciones y relaciones no lineales entre variables.")
    reporte.append("  Maneja datos heterogeneos con robustez.")
    reporte.append("  La importancia de variables ayuda a identificar los factores clave.")
    reporte.append("")
    reporte.append("Red Neuronal:")
    reporte.append("  Maxima capacidad predictiva con suficientes datos y features.")
    reporte.append("  Dificil de explicar (caja negra).")
    reporte.append("  Recomendada para alertas internas de mayor precision.")
    reporte.append("")
    reporte.append("RECOMENDACION PARA CONASET:")
    reporte.append("  - Regresion Multiple para reportes publicos (coeficientes explicables)")
    reporte.append("  - Red Neuronal para alertas internas (mayor precision)")
    reporte.append("  - Random Forest como baseline robusto intermedio")

    with open("reporte.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(reporte))
    print("  Reporte guardado como 'reporte.txt'")


# =====================================================================
# MENU PRINCIPAL
# =====================================================================

def main():
    print("\n" + "█" * 60)
    print("  TRABAJO DE REDES NEURONALES Y MODELOS PREDICTIVOS")
    print("  DATA SCIENCE")
    print("  Fuente: Observatorio CONASET (2000-2024)")
    print("█" * 60)

    verificar_api_conaset()
    df = cargar_datos()

    if "region" not in df.columns or "siniestros" not in df.columns:
        print("\n  Error: no se mapearon bien las columnas del Excel.")
        print("  Revisa las funciones de lectura y ajusta segun tu archivo.")
        return

    while True:
        print("\n" + "=" * 50)
        print("                  MENU PRINCIPAL")
        print("=" * 50)
        print("  [ Exploracion ]")
        print("   1. Ver datos (ultimas 10 filas)")
        print("   2. Estadistica descriptiva")
        print("   3. Graficos exploratorios")
        print("")
        print("  [ Modelos Clasicos ]")
        print("   4. Regresion Lineal Simple")
        print("   5. Regresion Multiple")
        print("   6. Random Forest")
        print("")
        print("  [ Red Neuronal ]")
        print("   7. Entrenar red neuronal")
        print("   8. Ver metricas red neuronal")
        print("   9. Ver graficos red neuronal")
        print("")
        print("  [ Comparativa ]")
        print("  10. Comparar todos los modelos")
        print("")
        print("  11. Salir")

        op = input("\n>> ").strip()

        if op == "1":
            print(df.tail(10).to_string())

        elif op == "2":
            mostrar_estadistica_descriptiva(df)

        elif op == "3":
            hacer_graficos_exploratorios(df)

        elif op == "4":
            _entrenar_simple(df)

        elif op == "5":
            _entrenar_multiple(df)

        elif op == "6":
            _entrenar_rf(df)

        elif op == "7":
            entrenar_red_neuronal(df)

        elif op == "8":
            mostrar_metricas_red()

        elif op == "9":
            graficar_resultados_red()

        elif op == "10":
            comparar_todos_los_modelos()

        elif op == "11":
            print("\n  Fin del trabajo. Cuidese estimado :)!!\n")
            break

        else:
            print("  Eso no es una opcion valida")


if __name__ == "__main__":
    main()
