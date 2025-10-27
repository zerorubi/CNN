# ===================================================
# Fases 4, 5 y 6 CRISP-DM - VERSIÓN RÁPIDA (30 min)
# Fashion-MNIST CNN - Optimizado para velocidad
# ===================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("🚀 INICIANDO IMPLEMENTACIÓN FASHION-MNIST CNN - VERSIÓN RÁPIDA")
print("⏰ Optimizado para ejecución en ~30 minutos")

# ===================================================
# FASE 4: MODELADO - CONSTRUCCIÓN DEL MODELO CNN RÁPIDO
# ===================================================

print("\n🔧 FASE 4: CONSTRUYENDO MODELO CNN OPTIMIZADO PARA VELOCIDAD...")

# Cargar Fashion-MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print(f"📐 Dimensiones del dataset:")
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

# PREPROCESAMIENTO RÁPIDO
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Expandir dimensiones para CNN (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"📊 Forma después del preprocesamiento:")
print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")

# Definir etiquetas de clases
class_names = ['Camiseta/Top', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']

# ===================================================
# VISUALIZACIÓN RÁPIDA
# ===================================================

print("\n📊 VISUALIZACIÓN RÁPIDA DEL DATASET...")

# Visualización rápida del dataset
plt.figure(figsize=(12, 8))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    idx = np.random.randint(len(x_train))
    plt.imshow(x_train[idx].squeeze(), cmap='gray')
    plt.title(f'{class_names[y_train[idx]]}', fontsize=9)
    plt.axis('off')
plt.suptitle('MUESTRAS ALEATORIAS - FASHION-MNIST', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Distribución de clases rápida
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
train_counts = np.bincount(y_train)
plt.bar(range(10), train_counts, color='skyblue', edgecolor='navy')
plt.title('Distribución - ENTRENAMIENTO', fontweight='bold')
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.xticks(range(10), [str(i) for i in range(10)])

plt.subplot(1, 2, 2)
test_counts = np.bincount(y_test)
plt.bar(range(10), test_counts, color='lightcoral', edgecolor='darkred')
plt.title('Distribución - TEST', fontweight='bold')
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.xticks(range(10), [str(i) for i in range(10)])

plt.tight_layout()
plt.show()

# ===================================================
# CONSTRUCCIÓN DEL MODELO CNN RÁPIDO
# ===================================================

def crear_modelo_rapido():
    model = keras.Sequential([
        # Bloque 1 - Capas convolucionales simples
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloque 2 - Capas convolucionales
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloque 3 - Capas finales
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),

        # Capas densas reducidas
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Crear y compilar modelo rápido
model = crear_modelo_rapido()

# Optimizador con learning rate más alto para convergencia más rápida
optimizer = Adam(learning_rate=0.0015)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ MODELO RÁPIDO CREADO Y COMPILADO")
model.summary()

# ===================================================
# FASE 5: EVALUACIÓN - ENTRENAMIENTO ACELERADO
# ===================================================

print("\n🎯 FASE 5: ENTRENAMIENTO ACELERADO (8 ÉPOCAS)...")

# Callbacks optimizados para velocidad
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=4,  # Reducido para terminar antes
    restore_best_weights=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,  # Reducción más agresiva
    patience=2,
    min_lr=1e-6,
    verbose=1
)

print("⏳ Iniciando entrenamiento acelerado...")

# Entrenar el modelo con configuración optimizada para velocidad
history = model.fit(
    x_train, y_train,
    batch_size=128,  # Batch size más grande para mayor velocidad
    epochs=8,        # Menos épocas
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
    shuffle=True
)

print("✅ ENTRENAMIENTO ACELERADO COMPLETADO")

# ===================================================
# VISUALIZACIÓN DEL ENTRENAMIENTO RÁPIDA
# ===================================================

print("\n📈 VISUALIZACIÓN RÁPIDA DEL ENTRENAMIENTO...")

plt.figure(figsize=(15, 5))

# Gráfico de precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión Entrenamiento',
         linewidth=2, marker='o', markersize=4, color='blue')
plt.plot(history.history['val_accuracy'], label='Precisión Validación',
         linewidth=2, marker='s', markersize=4, color='red')
plt.title('EVOLUCIÓN DE LA PRECISIÓN', fontsize=12, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True, alpha=0.3)

# Ajustar límites automáticamente
y_min_acc = min(min(history.history['accuracy']), min(history.history['val_accuracy']))
y_max_acc = max(max(history.history['accuracy']), max(history.history['val_accuracy']))
plt.ylim(max(0, y_min_acc - 0.05), min(1, y_max_acc + 0.05))

# Gráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida Entrenamiento',
         linewidth=2, marker='o', markersize=4, color='blue')
plt.plot(history.history['val_loss'], label='Pérdida Validación',
         linewidth=2, marker='s', markersize=4, color='red')
plt.title('EVOLUCIÓN DE LA PÉRDIDA', fontsize=12, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True, alpha=0.3)

# Ajustar límites automáticamente
y_min_loss = min(min(history.history['loss']), min(history.history['val_loss']))
y_max_loss = max(max(history.history['loss']), max(history.history['val_loss']))
plt.ylim(0, y_max_loss + 0.1)

plt.tight_layout()
plt.show()

# ===================================================
# FASE 6: DESPLIEGUE - EVALUACIÓN RÁPIDA
# ===================================================

print("\n🚀 FASE 6: EVALUACIÓN RÁPIDA Y PREDICCIONES...")

# Evaluación final del modelo
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"🎯 PRECISIÓN FINAL EN TEST: {test_accuracy:.4f}")

# Predicciones rápidas
y_pred = model.predict(x_test, verbose=0, batch_size=512)  # Batch grande para velocidad
y_pred_classes = np.argmax(y_pred, axis=1)

# ===================================================
# MATRIZ DE CONFUSIÓN RÁPIDA
# ===================================================

print("\n📊 MATRIZ DE CONFUSIÓN RÁPIDA...")

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)],
            annot_kws={'size': 8})
plt.title('MATRIZ DE CONFUSIÓN - FASHION-MNIST (MODELO RÁPIDO)',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('PREDICCIÓN', fontweight='bold')
plt.ylabel('VALOR REAL', fontweight='bold')
plt.tight_layout()
plt.show()

# Reporte de clasificación rápido
print("\n📋 REPORTE DE CLASIFICACIÓN RÁPIDO:")
print(classification_report(y_test, y_pred_classes,
                          target_names=[f'Clase {i}' for i in range(10)], digits=3))

# ===================================================
# ANÁLISIS RÁPIDO POR CLASE
# ===================================================

print("\n🔍 ANÁLISIS RÁPIDO DE PRECISIÓN POR CLASE:")

precision_por_clase = []
for i in range(10):
    mascara = (y_test == i)
    precision_clase = np.mean(y_pred_classes[mascara] == i)
    precision_por_clase.append(precision_clase)
    print(f"📊 {class_names[i]:<15}: {precision_clase:.3f}")

# Gráfico rápido de precisión por clase
plt.figure(figsize=(12, 5))
bars = plt.bar(range(10), precision_por_clase, color='lightgreen', edgecolor='darkgreen')
plt.title('PRECISIÓN POR CLASE - FASHION-MNIST', fontsize=14, fontweight='bold')
plt.xlabel('Clases')
plt.ylabel('Precisión')
plt.xticks(range(10), [str(i) for i in range(10)])
plt.grid(axis='y', alpha=0.3)

# Añadir valores en las barras
for bar, valor in zip(bars, precision_por_clase):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ===================================================
# DEMOSTRACIÓN RÁPIDA - PREDICCIONES EN TIEMPO REAL
# ===================================================

print("\n🎯 DEMOSTRACIÓN RÁPIDA - PREDICCIONES EN TIEMPO REAL")

def predecir_imagen_rapida(model, imagen):
    """Función rápida para predecir una imagen individual"""
    if len(imagen.shape) == 3:
        imagen = np.expand_dims(imagen, axis=0)

    prediccion = model.predict(imagen, verbose=0)
    clase_predicha = np.argmax(prediccion)
    confianza = np.max(prediccion)

    return clase_predicha, confianza

print("🔍 Probando con 6 imágenes aleatorias del test set:")

# Crear figura para mostrar las 6 imágenes
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i in range(6):
    idx = np.random.randint(len(x_test))
    imagen = x_test[idx]
    verdadero = y_test[idx]

    # Usar función de predicción rápida
    prediccion, confianza = predecir_imagen_rapida(model, imagen)

    # Determinar color según si es correcto o no
    es_correcto = (prediccion == verdadero)
    color = 'green' if es_correcto else 'red'
    simbolo = '✅' if es_correcto else '❌'

    # Mostrar la imagen
    axes[i].imshow(imagen.squeeze(), cmap='gray')

    # Título informativo
    titulo = (f'{simbolo} Real: {class_names[verdadero]}\n'
              f'Pred: {class_names[prediccion]}\n'
              f'Conf: {confianza:.3f}')

    axes[i].set_title(titulo, color=color, fontsize=11, fontweight='bold', pad=10)
    axes[i].axis('off')

plt.suptitle('🎯 PREDICCIONES RÁPIDAS - FASHION-MNIST CNN OPTIMIZADO',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# ===================================================
# PREDICCIONES CON ANÁLISIS DE CONFIANZA RÁPIDO
# ===================================================

print("\n🔍 PREDICCIONES CON ANÁLISIS DE CONFIANZA - 4 IMÁGENES")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i in range(4):
    idx = np.random.randint(len(x_test))
    imagen = x_test[idx]
    verdadero = y_test[idx]

    prediccion, confianza = predecir_imagen_rapida(model, imagen)
    es_correcto = (prediccion == verdadero)
    color = 'green' if es_correcto else 'red'

    # Mostrar imagen
    axes[i].imshow(imagen.squeeze(), cmap='gray')

    # Información detallada
    info_text = (f'Real: {class_names[verdadero]}\n'
                 f'Pred: {class_names[prediccion]}\n'
                 f'Confianza: {confianza:.3f}\n'
                 f'{"✓ CORRECTO" if es_correcto else "✗ ERROR"}')

    axes[i].set_title(info_text, color=color, fontsize=11, fontweight='bold', pad=10)
    axes[i].axis('off')

plt.suptitle('🔍 ANÁLISIS DE CONFIANZA - PREDICCIONES RÁPIDAS',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# ===================================================
# RESUMEN FINAL RÁPIDO
# ===================================================

print("\n" + "="*60)
print("🎉 RESUMEN FINAL - FASHION-MNIST CNN RÁPIDO")
print("="*60)
print(f"✅ Precisión final en test: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"✅ Pérdida final en test: {test_loss:.4f}")
print(f"✅ Épocas entrenadas: {len(history.history['accuracy'])}")
print(f"✅ Mejor precisión validación: {max(history.history['val_accuracy']):.4f}")
print(f"✅ Arquitectura: 5 capas Conv2D optimizadas")
print(f"✅ Tiempo estimado: ~15-25 minutos")
print(f"✅ Batch size: 128 (optimizado para velocidad)")

print("\n📊 MEJORES Y PEORES CLASES:")
# Ordenar clases por precisión
clases_ordenadas = sorted(zip(class_names, precision_por_clase),
                         key=lambda x: x[1], reverse=True)

print("   TOP 3 MEJORES:")
for i in range(3):
    print(f"      {clases_ordenadas[i][0]:<15}: {clases_ordenadas[i][1]:.3f}")

print("   TOP 3 PEORES:")
for i in range(3):
    print(f"      {clases_ordenadas[-(i+1)][0]:<15}: {clases_ordenadas[-(i+1)][1]:.3f}")

print("="*60)

print("\n🚀 IMPLEMENTACIÓN RÁPIDA COMPLETADA EXITOSAMENTE!")
print("⏰ Tiempo total estimado: 15-25 minutos")
