# 🔤 Reconocimiento de caracteres EMNIST (versión mejorada y corregida)
# ----------------------------------------------------------

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ----------------------------------------------------------
# 1️⃣ Cargar el dataset EMNIST (Balanced)
# ----------------------------------------------------------
ds, info = tfds.load('emnist/balanced', with_info=True, as_supervised=True)
train_ds, test_ds = ds['train'], ds['test']

# Convertir a numpy arrays
train_x = []
train_y = []
test_x = []
test_y = []

for image, label in tfds.as_numpy(train_ds):
    train_x.append(image)
    train_y.append(label)

for image, label in tfds.as_numpy(test_ds):
    test_x.append(image)
    test_y.append(label)

x1 = np.array(train_x)
y1 = np.array(train_y)
x2 = np.array(test_x)
y2 = np.array(test_y)

# ----------------------------------------------------------
# 2️⃣ Corregir orientación de las imágenes EMNIST
# ----------------------------------------------------------
x1 = np.rot90(x1, k=1, axes=(1, 2))
x1 = np.flip(x1, axis=2)

x2 = np.rot90(x2, k=1, axes=(1, 2))
x2 = np.flip(x2, axis=2)

# ----------------------------------------------------------
# 3️⃣ Normalización y preparación
# ----------------------------------------------------------
train_images = (x1 - 127.5) / 127.5
test_images = (x2 - 127.5) / 127.5

train_x = train_images.reshape(-1, 28, 28, 1)
test_x = test_images.reshape(-1, 28, 28, 1)

number_of_classes = len(np.unique(y1))
train_y = to_categorical(y1, number_of_classes)
test_y = to_categorical(y2, number_of_classes)

# ----------------------------------------------------------
# 4️⃣ Definir el modelo CNN mejorado
# ----------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # 🔥 previene sobreajuste
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(number_of_classes, activation='softmax')
])

# ----------------------------------------------------------
# 5️⃣ Compilar el modelo
# ----------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # 🔥 más estable
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------------------------------------
# 6️⃣ Callbacks
# ----------------------------------------------------------
MCP = ModelCheckpoint('mejor_modelo.keras', monitor='val_accuracy', save_best_only=True)
ES = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
RLP = ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.5)

# ----------------------------------------------------------
# 7️⃣ Entrenamiento
# ----------------------------------------------------------
history = model.fit(
    train_x, train_y,
    epochs=10,           # 🔼 aumentado para mejor precisión
    validation_data=(test_x, test_y),
    callbacks=[MCP, ES, RLP],
    batch_size=128,
    verbose=1
)

# ----------------------------------------------------------
# 8️⃣ Evaluación y prueba
# ----------------------------------------------------------
test_loss, test_acc = model.evaluate(test_x, test_y)
print(f"\n🔹 Precisión en test: {test_acc * 100:.2f}%")

# ----------------------------------------------------------
# 9️⃣ Visualización de resultados
# ----------------------------------------------------------
predictions = model.predict(test_x[:25])
pred_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_y[:25], axis=1)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(test_x[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {pred_labels[i]} / Real: {true_labels[i]}")
    plt.axis('off')
plt.show()
