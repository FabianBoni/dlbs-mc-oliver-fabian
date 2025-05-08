import os
from pathlib import Path
from PIL import Image

# TensorFlow-Loglevel auf WARN setzen, um JPEG-Warnungen zu unterdrücken
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Basisverzeichnisse
BASE_DIR  = Path('..') / 'data'
TRAIN_DIR = BASE_DIR / 'train'
VAL_DIR   = BASE_DIR / 'valid'
TEST_DIR  = BASE_DIR / 'test'

IMG_SIZE   = 224
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE

def remove_corrupt_images(folder: Path):
    """
    Durchsucht rekursiv alle Bilder im Ordner und
    löscht diejenigen, die sich nicht fehlerfrei laden lassen.
    """
    for ext in ('jpg', 'jpeg', 'png'):
        for img_path in folder.rglob(f'*.{ext}'):
            try:
                with Image.open(img_path) as img:
                    img.verify()
                # zusätzliches vollständiges Laden zum Abfangen weiterer Fehler
                with Image.open(img_path) as img2:
                    img2.load()
            except Exception:
                print(f"  ✗ Entferne defektes Bild {img_path}")
                img_path.unlink()

# Alle defekten Bilder aus train/val/test löschen
for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
    print(f"Cleaning folder: {d}")
    remove_corrupt_images(d)

def collect_paths_and_labels(directory: Path):
    """
    Sammelt alle Bildpfade und extrahiert Labels aus dem Dateinamen.
    """
    paths, labels = [], []
    for ext in ('jpg', 'jpeg', 'png'):
        for fn in directory.glob(f'*.{ext}'):
            paths.append(str(fn))
            labels.append(fn.stem.split('_')[0])
    return paths, labels

def preprocess_image(path, label):
    """
    Lädt das Bild, skaliert es mit Aspect-Ratio, paddet auf IMG_SIZE² und normiert auf [0,1].
    """
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    scale = tf.minimum(IMG_SIZE/tf.cast(h, tf.float32),
                       IMG_SIZE/tf.cast(w, tf.float32))
    new_h = tf.cast(tf.cast(h, tf.float32)*scale, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32)*scale, tf.int32)
    img = tf.image.resize(img, [new_h, new_w])
    img = tf.image.pad_to_bounding_box(
        img,
        (IMG_SIZE-new_h)//2,
        (IMG_SIZE-new_w)//2,
        IMG_SIZE,
        IMG_SIZE
    )
    img = img/255.0
    return img, label

def augment_image(image, label):
    """
    Führt horizontales Flip, zufällige Helligkeit und Kontrast durch.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label

def make_dataset(
    paths: list[str],
    labels: list[str],
    batch_size: int,
    *,
    cls2idx: dict[str,int] | None = None,
    augment: bool = False
):
    """
    Baut aus (paths, labels) ein gepuffertes und (optional) augmentiertes Dataset.
    Wenn cls2idx gegeben ist, wird es für One-Hot-Encoding wiederverwendet.
    """
    # Falls kein Mapping vorhanden, selbst erstellen
    if cls2idx is None:
        classes = sorted(set(labels))
        cls2idx = {c: i for i, c in enumerate(classes)}
    else:
        # Reihenfolge der Klassen konstant halten
        classes = sorted(cls2idx.keys())

    # Labels in Indizes umwandeln und one-hot kodieren
    y = [cls2idx[l] for l in labels]
    y = to_categorical(y, num_classes=len(classes))

    # Dataset zusammenbauen
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, cls2idx

# Pfade & Labels einlesen
train_paths, train_labels = collect_paths_and_labels(TRAIN_DIR)
val_paths,   val_labels   = collect_paths_and_labels(VAL_DIR)
test_paths,  test_labels  = collect_paths_and_labels(TEST_DIR)

# Einmal alle Klassen aus allen Splits sammeln, damit keine KeyError auftreten
all_labels = train_labels + val_labels + test_labels
classes = sorted(set(all_labels))
cls2idx = {c: i for i, c in enumerate(classes)}

# Trainings-Dataset + Mapping wiederverwenden
train_ds, _ = make_dataset(
    train_paths,
    train_labels,
    BATCH_SIZE,
    cls2idx=cls2idx,
    augment=True
)

# Validierungs- und Test-Datasets mit demselben Mapping
val_ds, _  = make_dataset(
    val_paths,
    val_labels,
    BATCH_SIZE,
    cls2idx=cls2idx
)
test_ds, _ = make_dataset(
    test_paths,
    test_labels,
    BATCH_SIZE,
    cls2idx=cls2idx
)

# Transfer-Learning mit MobileNetV2
base = MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=False, weights='imagenet')
base.trainable = True
# nur letzte 20 Layer werden feintrainiert
for layer in base.layers[:-20]:
    layer.trainable = False

inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(len(classes), activation='softmax')(x)
model = models.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training mit Callbacks
cbs = [
    callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2),
    callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=cbs
)

# Ergebnis-Plots
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],  label='train_acc')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'],      label='train_loss')
plt.plot(history.history['val_loss'],  label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.tight_layout()
plt.show()

# Endgültige Evaluation
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# Einige Beispiel-Vorhersagen
idx2cls = {v:k for k,v in cls2idx.items()}
for imgs, _ in test_ds.take(1):
    preds = model.predict(imgs)
    for i,p in enumerate(np.argmax(preds,axis=1)[:5]):
        print(f"Beispiel {i+1}: {idx2cls[p]}")