import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold

X = np.load("../Data/datasets/X_line_trawl.npy")
y = np.load("../Data/datasets/y_line_trawl.npy")
groups = np.load("../Data/datasets/groups_line_trawl.npy")

def evaluate_subset(y_true, y_pred, name="Subset"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{name} --------------------------------")
    print("Samples:", len(y_true))
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print(f"{name}: acc={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

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
cv_scores_total = []
cv_scores_line = []
cv_scores_trawl = []
i=1

for train_index, test_index in kf.split(X, y, groups=groups):
    print(f"Fold: {i} ===================================================")
    X_train_full, X_test = X[train_index], X[test_index]
    y_train_full, y_test = y[train_index], y[test_index]
    groups_train_full, groups_test = groups[train_index], groups[test_index]
    gear_test = np.array([g[0] for g in groups_test])
    line_mask = gear_test == "l"
    trawl_mask = gear_test == "t"

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X_train_full, y_train_full, groups=groups_train_full))

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]
    
    groups_train = groups_train_full[train_idx]
    groups_val = groups_train_full[val_idx]
    
    print("Train positive rate:", y_train.mean())
    print("Val positive rate:  ", y_val.mean())
    print("Test positive rate: ", y_test.mean())

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

    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)
    
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # Total
    total_metrics = evaluate_subset(y_test, y_pred, name=f"Fold {i} TOTAL")
    total_metrics["fold"] = i
    cv_scores_total.append(total_metrics)

    # Gear masks from groups_test
    gear_test = np.array([g[0] for g in groups_test])

    line_mask = gear_test == "l"
    trawl_mask = gear_test == "t"

    # Longliners
    if line_mask.sum() > 0:
        line_metrics = evaluate_subset(
            y_test[line_mask],
            y_pred[line_mask],
            name=f"Fold {i} LONGLINERS"
        )
        line_metrics["fold"] = i
        cv_scores_line.append(line_metrics)

    if trawl_mask.sum() > 0:
        trawl_metrics = evaluate_subset(
            y_test[trawl_mask],
            y_pred[trawl_mask],
            name=f"Fold {i} TRAWLERS"
        )
        trawl_metrics["fold"] = i
        cv_scores_trawl.append(trawl_metrics)

    i += 1

def print_summary(scores, name):
    accs = [s["accuracy"] for s in scores]
    precs = [s["precision"] for s in scores]
    recs = [s["recall"] for s in scores]
    f1s = [s["f1"] for s in scores]

    print(f"\n{name} SUMMARY ================================================")
    print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"Recall   : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"F1       : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

print_summary(cv_scores_total, "TOTAL CV")
print_summary(cv_scores_line, "LONGLINER CV")
print_summary(cv_scores_trawl, "TRAWLER CV")
