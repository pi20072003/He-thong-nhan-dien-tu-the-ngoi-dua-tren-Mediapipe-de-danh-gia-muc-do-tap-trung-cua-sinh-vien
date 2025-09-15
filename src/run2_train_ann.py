"""
Train ANN on MediaPipe Pose Landmarks (TF 2.15.0)

- Đọc data/pose_dataset.csv
- Chuẩn hoá (StandardScaler) -> lưu scaler
- Huấn luyện ANN (Keras) -> lưu models/posture_ann.h5
- Lưu mapping nhãn -> models/labels.json
"""

import os, json, pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_FILE = os.path.join("data", "pose_dataset.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "posture_ann.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

LABELS = {
    0: "ngoi thang",
    1: "guc dau",
    2: "nga nguoi",
    3: "quay trai",
    4: "quay phai",
    5: "chong tay",
}

def build_model(input_dim: int, n_classes: int = 10) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Không tìm thấy {DATA_FILE}. Hãy chạy run1_pose_capture.py để tạo dữ liệu.")

    df = pd.read_csv(DATA_FILE)
    y = df["label"].astype(int).values
    X = df.drop(columns=["label"]).values  # 33*(x,y,z,v)=132 đặc trưng

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    # Chuẩn hoá
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Lưu scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Xây & train model
    model = build_model(input_dim=X_train.shape[1], n_classes=len(LABELS))
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        verbose=1,
        callbacks=callbacks
    )

    # Đánh giá
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Lưu model & labels
    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False, indent=2)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved labels to {LABELS_PATH}")

if __name__ == "__main__":
    main()
