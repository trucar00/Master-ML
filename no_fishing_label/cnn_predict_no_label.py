import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

X = np.load("Data/datasets/X2.npy")
y = np.load("Data/datasets/y2.npy")
groups = np.load("Data/datasets/groups2.npy")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]

def build_model(input_shape):

    model = models.Sequential([
        layers.Input(shape=input_shape),

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

    return model


    
scaler = StandardScaler()
X_train = scaler.fit_transform(
    X_train.reshape(-1, X_train.shape[-1])
).reshape(X_train.shape)

X_test = scaler.transform(
    X_test.reshape(-1, X_test.shape[-1])
).reshape(X_test.shape)

classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight = dict(zip(classes, weights))

tf.keras.backend.clear_session()
model = build_model((X.shape[1], X.shape[2]))
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

# Now fit on unlabeled segments

# Load unlabeled segments
X_no_label = np.load("Data/datasets/X_no_label.npy")
print("No label: ", X_no_label.shape)

meta = np.load("Data/datasets/meta_no_label.npy", allow_pickle=True)

# Convert meta back to DataFrame (important!)
meta = pd.DataFrame(meta)
meta.columns = [
    "segment_id",
    "trajectory_id",
    "mmsi",
    "start_time",
    "end_time"
]

# Scale
X_no_label_scaled = scaler.transform(
    X_no_label.reshape(-1, X_no_label.shape[-1])
).reshape(X_no_label.shape)

# Predict
y_prob = model.predict(X_no_label_scaled).flatten()
y_pred = (y_prob >= 0.5).astype(int)

# Attach predictions to meta
meta["is_fishing"] = y_pred
meta["fishing_prob"] = y_prob

print(meta.head())

df = pd.read_csv("Data/trawl_segments_no_label.csv")

# Merge predictions onto every row in the segment
df = df.merge(
    meta[["segment_id", "is_fishing", "fishing_prob"]],
    on="segment_id",
    how="left"
)

print(df.head())
df.to_csv("Data/trawl_segments_with_predictions.csv", index=False)