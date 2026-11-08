# Credit Card Fraud Detection

Detección de pagos fraudulentos con tarjeta en comercio online, a partir de datos reales de la competición [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Kaggle, 2019).

> **Estado del proyecto:** ✅ Análisis exploratorio · ⏳ Feature engineering · ⏳ Modelado

## El problema

Cuando una tarjeta se pierde, se roba o se clona, puede usarse para compras online que su titular nunca hizo. Un sistema de detección de fraude tiene que equilibrar dos errores costosos:

- **Fraude no detectado (falso negativo):** el pago se aprueba y la empresa asume la pérdida.
- **Falsa alarma (falso positivo):** se bloquea a un cliente legítimo, que puede irse a la competencia.

El objetivo es construir un modelo que estime la probabilidad de fraude de cada transacción y permita elegir el equilibrio adecuado entre ambos errores.

## Los datos

Datos de Vesta Corporation, repartidos en dos tablas unidas por `TransactionID`:

| Tabla | Filas | Columnas | Contenido |
|---|---|---|---|
| `train_transaction` | 590.540 | 394 | Importe, tarjeta, dirección, email y variables anonimizadas. Incluye el target `isFraud` |
| `train_identity` | 144.233 | 41 | Dispositivo y conexión, en su mayoría anonimizados |

Los datos no se incluyen en el repositorio por su tamaño y licencia. Para reproducir el proyecto, descárgalos desde la [página de la competición](https://www.kaggle.com/competitions/ieee-fraud-detection/data) y descomprímelos en `data/raw/`.

## Principales hallazgos del análisis exploratorio

- **Desbalance:** solo el 3,5% de las transacciones son fraude. Un modelo que nunca detecte fraude tendría un 96,5% de accuracy, por lo que se evalúa con **PR-AUC**, recall y precisión.
- **Los nulos aportan información:** las transacciones sin dirección tienen cerca de un 12% de fraude, y las que incluyen datos de identidad, un 7,8% frente al 2,1% de las que no.
- **Ninguna variable explica el fraude por sí sola:** la mayor correlación con el target es de 0,15. La señal está repartida entre muchas variables y en sus combinaciones.
- **Relaciones no lineales e interacciones:** el importe tiene una relación en forma de U con el fraude que la correlación de Pearson no detecta. Al cruzarlo con los datos de identidad, se explica como la mezcla de dos comportamientos distintos.
- **El tiempo importa:** el fraude cambia a lo largo de los 182 días de datos y el conjunto de test es posterior, por lo que la validación será **temporal** en lugar de aleatoria.

El análisis completo está en [`notebooks/01_EDA.ipynb`](notebooks/01_EDA.ipynb).

## Estructura del repositorio

```
credit-fraud-detection/
├── data/
│   ├── raw/              # CSV originales de Kaggle (no incluidos)
│   └── processed/        # Datos limpios en Parquet, generados por el notebook 01
├── notebooks/
│   ├── 01_EDA.ipynb                  # Análisis exploratorio y limpieza
│   ├── 02_feature_engineering.ipynb  # (en desarrollo)
│   └── 03_modeling.ipynb             # (en desarrollo)
├── outputs/figures/      # Gráficos
├── requirements.txt
└── README.md
```

## Cómo reproducirlo

Requisitos: Python 3.11.

```bash
git clone https://github.com/pablofranb/credit-fraud-detection.git
cd credit-fraud-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Después, descarga los datos en `data/raw/` y ejecuta los notebooks en orden.

## Tecnologías

Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · Matplotlib · Seaborn · Jupyter

## Próximos pasos

- **Feature engineering:** indicadores de ausencia de datos, agrupación de categorías (sistema operativo, navegador, email), variables de tiempo y agregados por tarjeta, con separación temporal entre entrenamiento y validación.
- **Modelado:** comparación de un baseline, regresión logística, Random Forest, LightGBM y XGBoost mediante PR-AUC, elección del umbral de decisión e interpretación de las variables más importantes.

## Autor

Pablo Franco · [GitHub](https://github.com/pablofranb)
