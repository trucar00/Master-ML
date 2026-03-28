import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold

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

kf = StratifiedGroupKFold(n_splits=5, random_state=1, shuffle=True)
cv_scores = []
i=1

for train_index, val_index in kf.split(X_train, y_train, groups_train):
    print(f"Fold: {i} ===================================================")
    X_t, X_v = X_train[train_index], X_train[val_index]
    y_t, y_v = y_train[train_index], y_train[val_index]
    groups_t, groups_v = groups_train[train_index], groups_train[val_index]
    
    scaler = StandardScaler()
    X_t = scaler.fit_transform(
        X_t.reshape(-1, X_t.shape[-1])
    ).reshape(X_t.shape)

    X_v = scaler.transform(
        X_v.reshape(-1, X_v.shape[-1])
    ).reshape(X_v.shape)

    classes = np.unique(y_t)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_t
    )

    class_weight = dict(zip(classes, weights))

    tf.keras.backend.clear_session()
    model = build_model((X.shape[1], X.shape[2]))


    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_t, y_t,
        validation_data=(X_v, y_v),
        epochs=100,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1
    )

    y_prob = model.predict(X_v, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_v, y_pred)
    prec = precision_score(y_v, y_pred, zero_division=0)
    rec = recall_score(y_v, y_pred, zero_division=0)
    f1 = f1_score(y_v, y_pred, zero_division=0)

    print(confusion_matrix(y_v, y_pred))
    print(classification_report(y_v, y_pred, digits=4))
    print(f"Fold {i}: acc={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")

    cv_scores.append({
        "fold": i,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

    i += 1

accs = [s["accuracy"] for s in cv_scores]
precs = [s["precision"] for s in cv_scores]
recs = [s["recall"] for s in cv_scores]
f1s = [s["f1"] for s in cv_scores]

print("\nCV SUMMARY ================================================")
print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
print(f"Recall   : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
print(f"F1       : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

print("\nFINAL TRAIN ON DEV SET =====================================")

scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(
    X_train.reshape(-1, X_train.shape[-1])
).reshape(X_train.shape)

X_test_scaled = scaler_final.transform(
    X_test.reshape(-1, X_test.shape[-1])
).reshape(X_test.shape)

classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_final = dict(zip(classes, weights))

tf.keras.backend.clear_session()
final_model = build_model((X.shape[1], X.shape[2]))

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# use a small validation split inside dev for early stopping
history = final_model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=64,
    class_weight=class_weight_final,
    callbacks=[early_stop],
    verbose=1
)

print("\nFINAL TEST ================================================")

y_test_prob = final_model.predict(X_test_scaled, verbose=0).ravel()
y_test_pred = (y_test_prob >= 0.5).astype(int)

print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, digits=4))
print("Final test F1:", f1_score(y_test, y_test_pred))