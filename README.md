# Redes Neuronales CONASET

Trabajo de redes neuronales usando datos historicos oficiales de CONASET.

La idea es continuar el trabajo anterior de siniestros viales, pero esta vez
aplicando limpieza de datos y una red neuronal MLP, segun las ultimas guias del
profesor.

## Fuente de datos

- Excel oficial descargado desde el Observatorio CONASET:
  `Regionesdeocurrencia2000-2024.xlsx`
- Validacion de fuente publica:
  https://mapas-conaset.opendata.arcgis.com

## Que hace el codigo

El archivo principal es:

`red_neuronal_conaset.py`

El programa:

1. Valida la fuente publica de CONASET mediante ArcGIS.
2. Lee el Excel oficial, que viene con una hoja por anio.
3. Limpia datos nulos, filas de totales y columnas numericas.
4. Crea variables nuevas como tasas y datos historicos con `lag_1`.
5. Separa los datos en 80% para `fit` y 20% para `test`,
   respetando el orden temporal de los anios.
6. Normaliza con `StandardScaler` despues del split para evitar data leakage.
7. Entrena una red neuronal con capas `Dense`, activacion `ReLU` y `Dropout`.
8. Evalua con MAE, MSE, RMSE y R2.
9. Convierte el resultado a categorias bajo/medio/alto para calcular accuracy,
   precision, recall, F1-score y matriz de confusion.

## Red neuronal usada

```text
Input
Dense(64, ReLU)
Dropout(0.3)
Dense(32, ReLU)
Dropout(0.3)
Dense(16, ReLU)
Dense(1, ReLU)
```

Se usa ReLU en la salida porque la cantidad de siniestros no puede ser negativa.

## Como ejecutar

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar:

```bash
python red_neuronal_conaset.py
```

## Archivos importantes

- `red_neuronal_conaset.py`: codigo principal.
- `Regionesdeocurrencia2000-2024.xlsx`: datos originales CONASET.
- `requirements.txt`: librerias necesarias.

El programa muestra las metricas y graficos desde el menu, sin necesidad de
abrir archivos CSV aparte.

## Integrantes

- Rigo Vega
- Martin Caamano
- Favio Munoz
- Nikolas Maldonado
