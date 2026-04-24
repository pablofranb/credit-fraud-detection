# Credit Card Fraud Detection

Proyecto de análisis de datos y detección de fraude financiero usando el dataset IEEE-CIS de Kaggle.

## Objetivo
Detectar transacciones fraudulentas combinando análisis exploratorio, machine learning clásico y deep learning.

## Dataset
- **Fuente:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)
- **Tamaño:** ~590k transacciones, 434 features
- **Target:** `isFraud` (1 = fraude, 0 = legítimo)
## Conectarme
-- cd Desktop\credit-fraud-detection
-- venv\Scripts\activate
-- jupyter notebook


## Estructura del proyecto
credit-fraud-detection/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_ml.ipynb
│   └── 04_deep_learning.ipynb
├── outputs/figures/
├── requirements.txt
└── README.md

## Requerimientos
- Python 3.11 · Pandas · NumPy · Matplotlib · Seaborn
- Scikit-learn · XGBoost · LightGBM
- PyTorch
- dataset:
    - kaggle competitions download -c ieee-fraud-detection -p data/     
    - expandirlo:  Expand-Archive-Path data\ieee-fraud-detection.zip -DestinationPath data\


