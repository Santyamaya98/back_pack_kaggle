# Regression Project with LightGBM

This repository contains a regression project using the LightGBM model. The goal is to predict the price of a product based on various features. Throughout the project, we have implemented data preprocessing techniques, data transformation, and hyperparameter optimization to improve the model's performance.

## Project Description

1. **Data Preprocessing**:
   - A logarithmic transformation was applied to the target variable to stabilize variance and normalize the distribution.
   - Numerical features were scaled using `StandardScaler` to ensure all features are on the same scale.

2. **Regression Model**:
   - `LGBMRegressor` from LightGBM was used to train the regression model.
   - Cross-validation was implemented to evaluate the model's performance.

3. **Hyperparameter Optimization**:
   - Optuna was used to optimize the model's hyperparameters, including the number of estimators, learning rate, and number of leaves.
   - `early_stopping` was implemented to prevent overfitting during training.

4. **Model Evaluation**:
   - The model was evaluated using the Root Mean Squared Error (RMSE) to measure prediction accuracy.
   - A residual plot was generated to check for homoscedasticity in the data.

## Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `lightgbm`, `optuna`, `scikit-learn`, `matplotlib`, `seaborn`

## Usage Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   
# Proyecto de Regresión con LightGBM

Este repositorio contiene un proyecto de regresión utilizando el modelo LightGBM. El objetivo es predecir el precio de un producto basado en varias características. A lo largo del proyecto, hemos implementado técnicas de preprocesamiento, transformación de datos, y optimización de hiperparámetros para mejorar el rendimiento del modelo.

## Descripción del Proyecto

1. **Preprocesamiento de Datos**:
   - Se aplicó una transformación logarítmica a la variable objetivo para estabilizar la varianza y normalizar la distribución.
   - Se escalaron las características numéricas utilizando `StandardScaler` para asegurar que todas las características tengan la misma escala.

2. **Modelo de Regresión**:
   - Se utilizó `LGBMRegressor` de LightGBM para entrenar el modelo de regresión.
   - Se implementó la validación cruzada para evaluar el rendimiento del modelo.

3. **Optimización de Hiperparámetros**:
   - Se utilizó Optuna para optimizar los hiperparámetros del modelo, incluyendo el número de estimadores, la tasa de aprendizaje, y el número de hojas.
   - Se implementó el `early_stopping` para evitar el sobreajuste durante el entrenamiento.

4. **Evaluación del Modelo**:
   - Se evaluó el modelo utilizando el error cuadrático medio (RMSE) para medir la precisión de las predicciones.
   - Se generó un gráfico de residuos para verificar la homocedasticidad de los datos.

## Requisitos

- Python 3.x
- Librerías: `numpy`, `pandas`, `lightgbm`, `optuna`, `scikit-learn`, `matplotlib`, `seaborn`

## Instrucciones de Uso

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/tu-repositorio.git