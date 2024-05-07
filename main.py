import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics

# Función para calcular sensibilidad y especificidad
def calcular_metricas_confusion(y_true, y_pred):
    matriz_confusion = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = matriz_confusion.ravel()
    sensibilidad = tp / (tp + fn)
    especificidad = tn / (tn + fp)
    return sensibilidad, especificidad

# Establecer el estilo seaborn
sns.set()

# Leer datos
datos = pd.read_csv('diabetes.csv')
print("Muestra:", len(datos))
print("Nulos\n", datos.isna().sum())

X1 = datos[['Glucose','DiabetesPedigreeFunction','Pregnancies', 'Insulin','SkinThickness']]
Y1 = datos['Outcome']

# Estandarizar los datos
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)

# Dividir el conjunto de datos en entrenamiento y prueba para el modelo de red neuronal
X_train_nn, X_test_nn, Y_train_nn, Y_test_nn = train_test_split(X1_scaled, Y1, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de red neuronal
model_nn = Sequential()
model_nn.add(Dense(128, input_dim=5, activation='relu'))
model_nn.add(Dense(32, activation='relu'))
model_nn.add(Dense(8, activation='relu'))
model_nn.add(Dense(1, activation='sigmoid'))
model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_nn = model_nn.fit(X_train_nn, Y_train_nn, validation_split=0.33, epochs=40, batch_size=10, verbose=0)

# Obtener probabilidades predichas para la red neuronal
probs_nn = model_nn.predict(X1_scaled)

# Calcular la curva ROC para la red neuronal
fpr_nn, tpr_nn, _ = roc_curve(Y1, probs_nn)
roc_auc_nn = auc(fpr_nn, tpr_nn)

# Trazar la curva ROC para la red neuronal
plt.figure()
lw = 2
plt.plot(fpr_nn, tpr_nn, color='darkorange', lw=lw, label='ROC curve NN (area = %0.2f)' % roc_auc_nn)

#-------REG LOG AREA------------

# Dividir el conjunto de datos en entrenamiento y prueba para la regresión logística
X_train_lr, X_test_lr, Y_train_lr, Y_test_lr = train_test_split(X1_scaled, Y1, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de regresión logística
model_lr = LogisticRegression()
model_lr.fit(X_train_lr, Y_train_lr)

# Calcular el área bajo la curva ROC para la regresión logística
probs_lr = model_lr.predict_proba(X_test_lr)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(Y_test_lr, probs_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Trazar la curva ROC para la regresión logística
plt.plot(fpr_lr, tpr_lr, color='green', lw=lw, label='ROC curve LR (area = %0.2f)' % roc_auc_lr)

#-------RANDOM FOREST------------

# Dividir el conjunto de datos en entrenamiento y prueba para Random Forest
X_train_rf, X_test_rf, Y_train_rf, Y_test_rf = train_test_split(X1_scaled, Y1, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_rf, Y_train_rf)

# Calcular el área bajo la curva ROC para Random Forest
probs_rf = model_rf.predict_proba(X_test_rf)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(Y_test_rf, probs_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Trazar la curva ROC para Random Forest
plt.plot(fpr_rf, tpr_rf, color='blue', lw=lw, label='ROC curve RF (area = %0.2f)' % roc_auc_rf)




#-------SVM------------

# Dividir el conjunto de datos en entrenamiento y prueba para SVM
X_train_svm, X_test_svm, Y_train_svm, Y_test_svm = train_test_split(X1_scaled, Y1, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de SVM
model_svm = SVC(kernel='linear', probability=True, random_state=42)
model_svm.fit(X_train_svm, Y_train_svm)

# Calcular el área bajo la curva ROC para SVM
probs_svm = model_svm.predict_proba(X_test_svm)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(Y_test_svm, probs_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Trazar la curva ROC para SVM
plt.plot(fpr_svm, tpr_svm, color='pink', lw=lw, label='ROC curve SVM (area = %0.2f)' % roc_auc_svm)

# Añadir detalles al gráfico ROC
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('Accuracy of Models')
plt.show()

# Graficar historial de precisión y pérdida de la red neuronal
plt.plot(history_nn.history['accuracy'])
plt.plot(history_nn.history['val_accuracy'])
plt.title('Model Accuracy of Neural Network')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history_nn.history['loss'])
plt.plot(history_nn.history['val_loss'])
plt.title('Model Loss of Neural Network')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#AREA DE EFICACIA DE LA NEURAL NETWORK

# Calcular métricas de evaluación para la red neuronal
y_pred_nn = (model_nn.predict(X_test_nn) > 0.5).astype("int32")
error_general_nn = 1 - metrics.accuracy_score(Y_test_nn, y_pred_nn)
acierto_nn = metrics.accuracy_score(Y_test_nn, y_pred_nn)
sensibilidad_nn, especificidad_nn = calcular_metricas_confusion(Y_test_nn, y_pred_nn)

print("Eficacia de la Neural Network:")
print("Grado de error general en los datos de prueba:", error_general_nn)
print("Porcentaje de acierto en los datos de prueba:", acierto_nn)
print("Sensibilidad:", sensibilidad_nn)
print("Especificidad:", especificidad_nn)
print("Área bajo la curva ROC:", roc_auc_nn)



#new

# Realizar predicciones en el conjunto de prueba para la regresión logística
Y_pred_lr = model_lr.predict(X_test_lr)

# Calcular el grado de error general en los datos de prueba para la regresión logística
error_general_lr = 1 - metrics.accuracy_score(Y_test_lr, Y_pred_lr)

# Calcular el porcentaje de acierto en los datos de prueba para la regresión logística
acierto_lr = metrics.accuracy_score(Y_test_lr, Y_pred_lr)

# Calcular las métricas de matriz de confusión para la regresión logística
confusion_matrix_lr = metrics.confusion_matrix(Y_test_lr, Y_pred_lr)
verdaderos_positivos_lr = confusion_matrix_lr[1, 1]
falsos_negativos_lr = confusion_matrix_lr[1, 0]
falsos_positivos_lr = confusion_matrix_lr[0, 1]
verdaderos_negativos_lr = confusion_matrix_lr[0, 0]

# Calcular sensibilidad y especificidad para la regresión logística
sensibilidad_lr = verdaderos_positivos_lr / (verdaderos_positivos_lr + falsos_negativos_lr)
especificidad_lr = verdaderos_negativos_lr / (verdaderos_negativos_lr + falsos_positivos_lr)

# Calcular el área bajo la curva ROC para la regresión logística
probs_lr = model_lr.predict_proba(X_test_lr)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(Y_test_lr, probs_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Imprimir resultados para la regresión logística
print("Eficacia de la Regresión Logística:")
print("Grado de error general en los datos de prueba:", error_general_lr)
print("Porcentaje de acierto en los datos de prueba:", acierto_lr)
print("Sensibilidad:", sensibilidad_lr)
print("Especificidad:", especificidad_lr)
print("Área bajo la curva ROC:", roc_auc_lr)
