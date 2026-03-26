import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

X = np.load("Data/datasets/X.npy")
y = np.load("Data/datasets/y.npy")

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(
    X_train.reshape(-1, X_train.shape[-1])
).reshape(X_train.shape)

X_val = scaler.transform(
    X_val.reshape(-1, X_val.shape[-1])
).reshape(X_val.shape)

classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight = dict(zip(classes, weights))

model = models.Sequential([
    layers.Input(shape=(X.shape[1], X.shape[2])),  # (11, 5)

    layers.Conv1D(filters=256, kernel_size=3, padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.Conv1D(filters=256, kernel_size=3, padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.Conv1D(filters=256, kernel_size=3, padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.Conv1D(filters=256, kernel_size=3, padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.Conv1D(filters=256, kernel_size=3, padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

y_prob = model.predict(X_val)
y_pred = (y_prob >= 0.5).astype(int).ravel()

print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred, digits=4))
print("F1:", f1_score(y_val, y_pred))

# K-fold CV