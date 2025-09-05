"""
Realtime Posture Classification (TF 2.15.0 + MediaPipe 0.10.21)

- Mở webcam, lấy landmarks
- Chuẩn hoá với scaler.pkl
- Dự đoán bằng posture_ann.h5
- Hiển thị nhãn & độ tự tin (prob)
"""

import os, json, pickle, cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "posture_ann.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# Load model, scaler, labels
if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, LABELS_PATH]):
    raise FileNotFoundError("Thiếu model/scaler/labels. Hãy chạy run2_train_ann.py trước.")

model = load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = {int(k): v for k, v in json.load(f).items()}

# Mediapipe
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def predict_label(features_132: np.ndarray) -> tuple[str, float]:
    """features_132 shape: (132,) -> returns (label_text, prob)"""
    X = scaler.transform(features_132.reshape(1, -1))
    probs = model.predict(X, verbose=0)[0]
    cls_id = int(np.argmax(probs))
    return LABELS.get(cls_id, "unknown"), float(probs[cls_id])

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam.")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok: break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                # Vẽ skeleton
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Rút đặc trưng: 33 * (x,y,z,visibility) = 132
                feats = []
                for lm in res.pose_landmarks.landmark:
                    feats += [lm.x, lm.y, lm.z, lm.visibility]
                feats = np.array(feats, dtype=np.float32)

                # Dự đoán
                label_text, prob = predict_label(feats)
                cv2.putText(frame, f"{label_text} ({prob:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No pose detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow("Posture Classification (ANN)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
