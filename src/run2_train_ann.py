"""
Train ANN on MediaPipe Pose Landmarks (Optimized for Small Dataset)
- Giảm overfitting mạnh: BatchNorm, L2 regularization, Dropout hợp lý, model nhỏ gọn
- Báo cáo đầy đủ: Accuracy, Precision, Recall, F1, Confusion Matrix
Cần cài đặt:
    pip install tensorflow==2.15.0
    pip install scikit-learn==1.5.0
    pip install seaborn==0.13.2
    pip install matplotlib==3.8.2
"""

import os, json, pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ===== PATH =====
DATA_FILE = os.path.join("data", "pose_dataset.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "posture_ann.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# ===== LABELS =====
LABELS = {
    0: "ngoi thang",
    1: "guc dau",
    2: "nga nguoi",
    3: "quay trai",
    4: "quay phai",
    5: "chong tay",
}

# ===== MODEL =====
def build_model(input_dim: int, n_classes: int = 6) -> keras.Model:
    L2 = regularizers.l2(1e-4)

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation="relu", kernel_regularizer=L2),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(64, activation="relu", kernel_regularizer=L2),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation="relu", kernel_regularizer=L2),

        layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("Không tìm thấy pose_dataset.csv – hãy chạy run1 trước!")

    df = pd.read_csv(DATA_FILE)
    y = df["label"].astype(int).values
    X = df.drop(columns=["label"]).values

    # ===== TRAIN/VAL SPLIT =====
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ===== SCALING =====
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # ===== BUILD MODEL =====
    model = build_model(input_dim=X_train.shape[1], n_classes=len(LABELS))

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience=6, factor=0.5, verbose=1),
        keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, verbose=1)
    ]

    # ===== TRAIN =====
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=120,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    # ===== EVALUATION =====
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Accuracy: {val_acc:.4f}")

    y_pred = np.argmax(model.predict(X_val), axis=1)

    print("\n====== ACCURACY ======")
    print("Accuracy:", accuracy_score(y_val, y_pred))

    print("\n====== PRECISION - RECALL - F1 ======")
    print(classification_report(
        y_val, y_pred,
        target_names=list(LABELS.values()),
        digits=4
    ))

    print("\n====== CONFUSION MATRIX ======")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)

    # ===== DRAW CONFUSION MATRIX =====
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=list(LABELS.values()),
        yticklabels=list(LABELS.values()),
        cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    # ===== SAVE MODEL =====
    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False, indent=2)

    print("\nSaved: model, scaler, labels, confusion_matrix.png")
        # ===== SAVE METRICS TO FILE =====
    report_path = os.path.join(MODEL_DIR, "metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        
        f.write("===== MODEL EVALUATION REPORT =====\n\n")

        f.write(f"Validation Accuracy: {val_acc:.4f}\n\n")

        f.write("====== ACCURACY ======\n")
        f.write(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}\n\n")

        f.write("====== PRECISION - RECALL - F1 ======\n")
        f.write(classification_report(
            y_val, y_pred,
            target_names=list(LABELS.values()),
            digits=4
        ))
        f.write("\n")

        f.write("====== CONFUSION MATRIX ======\n")
        f.write(np.array2string(cm))
        f.write("\n\n")

        f.write("====== LABEL MAPPING ======\n")
        for k, v in LABELS.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("====== TRAINING PARAMETERS ======\n")
        f.write(f"Epochs: {len(history.history['loss'])}\n")
        f.write("Batch size: 16\n")
        f.write("Optimizer: Adam (lr=0.001)\n")
        f.write("Loss: sparse_categorical_crossentropy\n\n")

        f.write("====== MODEL ARCHITECTURE ======\n")
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    print(f"\nSaved metrics report to: {report_path}\n")


if __name__ == "__main__":
    main()
