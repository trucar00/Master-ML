import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

gear = "t_new"

X = np.load(f"datasets/X_{gear}.npy")
y = np.load(f"datasets/y_{gear}.npy")
groups = np.load(f"datasets/groups_{gear}.npy")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

train_idx, val_idx = next(gss.split(X, y, groups=groups))

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
groups_train, groups_val = groups[train_idx], groups[val_idx]

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
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

best_epoch = len(history.history["loss"])
print("Best epoch:", best_epoch)

# Now fit on the whole training set

scaler_final = StandardScaler()
X_all_scaled = scaler_final.fit_transform(
    X.reshape(-1, X.shape[-1])
).reshape(X.shape)

classes_all = np.unique(y)
weights_all = compute_class_weight(
    class_weight="balanced",
    classes=classes_all,
    y=y
)
class_weight_all = dict(zip(classes_all, weights_all))

tf.keras.backend.clear_session()
final_model = build_model((X.shape[1], X.shape[2]))

final_model.fit(
    X_all_scaled, y,
    epochs=best_epoch,
    batch_size=64,
    class_weight=class_weight_all,
    verbose=1
)

# Load unlabeled segments
X_no_label = np.load(f"datasets/X_{gear}_no_label.npy")
print("No label: ", X_no_label.shape)

meta = np.load(f"datasets/meta_{gear}_no_label.npy", allow_pickle=True)

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
X_no_label_scaled = scaler_final.transform(
    X_no_label.reshape(-1, X_no_label.shape[-1])
).reshape(X_no_label.shape)

# Predict
y_prob = final_model.predict(X_no_label_scaled).flatten()
y_pred = (y_prob >= 0.5).astype(int)

# Attach predictions to meta
meta["is_fishing"] = y_pred
meta["fishing_prob"] = y_prob

print(meta.head())

df = pd.read_csv(f"../Data/{gear}_segments_no_label.csv")

# Merge predictions onto every row in the segment
df = df.merge(
    meta[["segment_id", "is_fishing", "fishing_prob"]],
    on="segment_id",
    how="left"
)

print("Shape of df:", df.shape) # should be: (114917, 12)
print(df.head())

df.to_csv(f"{gear}_segments_with_predictions.csv", index=False)