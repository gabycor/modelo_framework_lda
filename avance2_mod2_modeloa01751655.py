# -*- coding: utf-8 -*-

# Librerías y lectura de datos

"""Librerías"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, KFold


"""Lectura de datos

Se hace la lectura de datos "dataset.csv". Fuente: https://www.kaggle.com/datasets/naveenkumar20bps1137/predict-students-dropout-and-academic-success
"""

df = pd.read_csv('dataset.csv')
df.head()

"""Lectura de datos
Se hace la lectura de datos "dataset.csv". Fuente: https://www.kaggle.com/datasets/naveenkumar20bps1137/predict-students-dropout-and-academic-success
"""
df = pd.read_csv('dataset.csv')

"""Preprocesamiento de datos
Forma del dataset
"""

# Tipo de datos
#df.info()

#df.describe().round(3)

# Para hacer una división de las columnas que son numéricas y categóricas, se hace desde el principio el select de las columnas
categoricas = df.select_dtypes(include='object').columns
numericas = df.select_dtypes(include='number').columns

"""Tratamiento de datos faltantes"""

# Conteo de las variables faltantes
#df.isnull().sum()/df.shape[0]*100 #porcentaje
# Se observa que no hay datos faltantes

# Se verifica si hay datos duplicados
#df.duplicated().sum()
# se puede observar que no hay datos duplicados por lo que no hay que tratar los datos en cuestión

"""Exploración de datos"""

# Observamos la variable a predecir para observar su distribución
#df['Target'].value_counts()

# Descomentar la siguiente línea de código para poder graficar su distribución observando si se trata de una distribucón sesgada o simétrica
# sns.countplot(df, x='Target')

# Debido a que solo nos interesa si el estudiante se gradúa o no eliminamos "Enrolled"
# Eesto debido a que el estudiante todavía puede seguir o dejar la escuela
#df=df[df.Target!='Enrolled']

# Descomentar la siguiente línea de código para poder graficar nuevamente la distribución de "Target" con los nuevos valores
# sns.countplot(df, x='Target')

"""Tratamiento de datos"""

# Cambiamos los valores categóricos a numéricos
# Se hace el cambio únicamente de "target"

# One-hot encode
df = pd.get_dummies(df, columns=categoricas)

# Se tira uno de las columnas resultantes de "Target" para solo quedarnos con la columna con resultados binarios "Target_Dropout"
df.drop("Target_Graduate", axis=1, inplace=True)

"""Correlaciones"""

# Se observan las correlaciones existentes
matriz_correlacion = df.corr()

"""Eliminación de columnas irrelevantes para el modelo"""

# Función para que nos de los pares de columnas altamente relacionados
umbral = 0.7

alto_umbral = []

for col in matriz_correlacion.columns:
    correlated_cols = matriz_correlacion.index[matriz_correlacion[col] >= umbral].tolist()
    correlated_cols.remove(col)
    for correlated_col in correlated_cols:
        pair = (col, correlated_col)
        alto_umbral.append(pair)

#print(alto_umbral)

# Se decide eliminar las columnas que están altamente correlacionadas en este caso se hace un estudio del par de variables correlacionadas para tomar la decisión
columnas_a_eliminar = ['Nacionality', 'Curricular units 1st sem (enrolled)', 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 1st sem (grade)','Curricular units 2nd sem (enrolled)',  'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)']

# Eliminar las columnas especificadas
# Se elimina Nacionalidad debido a que se prefiere quedar con una menor granularidad para tener un mayor nivel de explicatividad del modelo (se queda con Internacionalidad 1 en internacional, 0 portugal)
# Se elimina Curricular units 1st sem (enrolled) debido a que no nos interesa lo que se metió sino lo que se cumplió
# Se elimina Curricular units 2nd sem (credited) debido a que ya hicimos la relación anterior y por lo mismo se decidió elegir
# Al estar relacionaod con el de primer semestre se elimina el 'Curricular units 2nd sem (evaluations)'
df = df.drop(columnas_a_eliminar, axis=1)

# Selección de los datos target
target = df['Target_Dropout']

# Paso 1 del split: training and testing sets
x_train,x_test,y_train,y_test = train_test_split(df.drop('Target_Dropout', axis =1), target, test_size=0.2, random_state=42)

# Paso 2 del split: training and val sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

"""Entrenamiento de modelo"""

# Train
# Se hace uso del Análisis discriminante lineal para entrenar el modelo
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)

# Genera predicciones para test y validation
y_pred_test = lda.predict(x_test)
y_pred_val = lda.predict(x_val)

"""Evaluación del modelo (test) - básico

"""

# Test
# Calculo e impresión del accuracy obtenido por la predicción de test
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f'Precisión en el conjunto de test: {accuracy_test}')

# Impresión de matriz de confusión para observar el desempeño del modelo (test)
confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
print("\nMatriz de Confusión (test):")
print(confusion_matrix_test)

# Impresión del reporte de clasificación para observar el desempeño del modelo (test)
classification_report_test = classification_report(y_test, y_pred_test)
print("\nReporte de Clasificación (test):")
print(classification_report_test)

"""Evaluación del modelo (train/test) - cross validation"""

# Ajusta el modelo lda para train
lda.fit(x_train, y_train)

# Calculo de scores para train con cross validation
scores = cross_val_score(estimator=lda, X=x_train, y=y_train, cv=10)
mean_score = scores.mean()
std_score = scores.std()

# Se imprime del calculo el score de media y desviación estándar para el train
print("Mean Score:", mean_score)
print("Standard Deviation Score:", std_score)

# Se imprime el score del set de test
test_score = lda.score(x_test, y_test)
print("Test Set Score:", test_score)

# Se grafica un histograma para poder observar como se ven los scores resultantes del cross validation del train
plt.figure(figsize=(8, 6))
plt.bar(range(len(scores)), scores)
plt.xlabel("Fold")
plt.ylabel("Cross-Validation Score")
plt.title("Cross-Validation Scores for LDA")
plt.show()

# Se realiza un box plot para poder visualizar como se encuentran los scores
plt.figure(figsize=(8, 6))
sns.boxplot(x=scores)
plt.xlabel("Cross-Validation Score")
plt.title("Box Plot of Cross-Validation Scores for LDA")
plt.show()
# Como resultado del histograma y boxplot se puede decir que los scores generados por el dataset de train no tiene demasiada variabilidad en sus datos.
# Como resultado de hacer cross se tiene que los score están entre 0.86 y 0.89 aproximadamente, de igual manera se observa que no hay outliers visibles. (todo con el dataset de train)

"""Evaluación del modelo (train/test) - cross validation, gridsearch"""

# LDA
lda = LinearDiscriminantAnalysis()

# Definir los hiperparámetros para buscar. Con la siguiente línea de código se busca probar cualquier parámetro para ver cual funciona mejor.
# svd = singular value decomposition (no acepta relu).
# lsqr = least squares solution para sistema de ecuaciones lineas.
# eigen = componentes discriminantes de la matriz de dispoersión.
# N_components = parámetros a calcular
params = {'solver': ['svd', 'lsqr', 'eigen'], 'n_components': [1, 2, 3]}

# Uso de validación cruzada para ajustar los hiperparámetros en el conjunto de entrenamiento
search = GridSearchCV(lda, params, cv=5, scoring='accuracy')
search.fit(x_train, y_train)

# Mejores hiperparámetros y puntuación en el conjunto de (train)
print("Mejores hiperparámetros:", search.best_params_)
print("Mejor puntuación en entrenamiento:", search.best_score_)

# Evaluación del modelo con los mejores hiperparámetros en el conjunto de (train)
best_lda = search.best_estimator_
test_score = best_lda.score(x_test, y_test)
print("Puntuación en prueba:", test_score)
# Las impresiones nos muestran los mejores parámetros junto con las puntuaciones en train y test.

"""Evaluación del modelo (validation) - con los mejores parámetros obtenidos"""

# Se divide aleatorioamente el dataset en 7 pliegues con aleatoriedad
kf = KFold(n_splits=7, shuffle=True, random_state=42)

# se genera el entrenamiento de cada split
mse_scores = []
for train_index, test_index in kf.split(x_val):
    X_train, X_test = x_val.iloc[train_index], x_val.iloc[test_index]
    y_train, y_test = y_val.iloc[train_index], y_val.iloc[test_index]

    # Entrena tu modelo en X_train, y_train
    model = LinearDiscriminantAnalysis(n_components=1, solver='svd')
    model.fit(X_train, y_train)

    # Realiza predicciones en X_test
    y_pred = model.predict(X_test)

    # Calcula el error cuadrático medio en este pliegue (opcional)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

# Se grafican los errores cuadráticos para obtener comparativos de como funciona el modelo para una división de datasets de validation.
plt.figure(figsize=(10, 5))
plt.plot(range(1, 8), mse_scores, marker='o', linestyle='-', color='b')
plt.title('Errores Cuadráticos Medios (MSE) en Validación Cruzada de 7 Pliegues')
plt.xlabel('Pliegues')
plt.ylabel('MSE')
plt.xticks(np.arange(1, 8))
plt.grid(True)
plt.show()
# se observa que no hay cambios significativos entre las validaciones por lo que se puede decir que la varianza es baja. Se desarrolla cada punto con mayor profundidad en el reporte.

# Se grafican los errores cuadráticos para obtener comparativos de como funciona el modelo para una división de datasets de validation pero ahora en histograma.
plt.figure(figsize=(8, 6))
plt.bar(range(len(mse_scores)), mse_scores)
plt.xlabel("Fold")
plt.ylabel("Cross-Validation Score")
plt.title("Cross-Validation Scores for LDA")
plt.show()