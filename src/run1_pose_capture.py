"""
Pose Capture Optimized for Sitting Posture (Laptop/PC)
- Nhấn S để bắt đầu lấy mẫu
- Nhấn E để dừng lấy mẫu
- Lọc landmark upper-body
- Hiển thị số mẫu từng tư thế
Cần cài đặt:
    pip install mediapipe==0.10.21
    pip install protobuf==4.25.3
    pip install opencv-python==4.11.0
    pip install pandas==2.3.1
    pip install numpy==1.26.4

"""

import os, csv, cv2, time
import mediapipe as mp
import pandas as pd

# ===== PATH =====
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_FILE = os.path.join(DATA_DIR, "pose_dataset.csv")

# ===== LABELS =====
LABELS = {
    0: "ngoi thang",
    1: "guc dau",
    2: "nga nguoi",
    3: "quay trai",
    4: "quay phai",
    5: "chong tay",
}

# ===== UPPER BODY LANDMARKS =====
UPPER_BODY = [
    0, 1, 2, 3, 4,      # Head + nose + eyes + ears
    11, 12,            # Shoulders
    13, 14,            # Elbows
    15, 16             # Wrists
]

# ===== CSV HEADER =====
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        w = csv.writer(f)
        header = ["label"]
        for i in range(33):
            header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
        w.writerow(header)

# ===== COUNT SAMPLES =====
def count_samples():
    if not os.path.exists(CSV_FILE):
        return [0] * len(LABELS)
    df = pd.read_csv(CSV_FILE)
    cnt = df["label"].value_counts().to_dict()
    return [cnt.get(i, 0) for i in LABELS]

sample_counts = count_samples()

# ===== SAVE ROW =====
def save_row(label_id, pose_landmarks):
    row = [label_id]
    for lm in pose_landmarks.landmark:
        row += [lm.x, lm.y, lm.z, lm.visibility]

    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)

    sample_counts[label_id] += 1


# ===== MAIN LOOP =====
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam.")

    current_label = 0
    collecting = False    # <---- TRẠNG THÁI LẤY MẪU
    last_save_time = 0
    last_vec = None

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    print("0–5 đổi nhãn | S bắt đầu thu | E dừng thu | ESC/Q thoát")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            # ---- DRAW ----
            if res.pose_landmarks:
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ---- HUD ----
            cv2.putText(frame, f"Label [{current_label}] : {LABELS[current_label]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,255,0) if collecting else (0,150,255), 2)

            cv2.putText(frame, f"Collecting: {'YES' if collecting else 'NO'}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0) if collecting else (0,255,255), 2)

            y_offset = 100
            for i, name in LABELS.items():
                cv2.putText(frame, f"{i}-{name}: {sample_counts[i]} mẫu",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255,255,255), 1)
                y_offset += 25

            cv2.imshow("Pose Capture PRO", frame)

            # ---- READ KEY ----
            key = cv2.waitKey(10) & 0xFF

            # Chọn nhãn 0–5
            if ord("0") <= key <= ord("5"):
                current_label = key - ord("0")

            # Bắt đầu thu mẫu
            if key in [ord('s'), ord('S')]:
                collecting = True
                print("=== BẮT ĐẦU THU MẪU ===")

            # Dừng thu mẫu
            if key in [ord('e'), ord('E')]:
                collecting = False
                print("=== DỪNG THU MẪU ===")

            # Thoát
            if key in [27, ord('q'), ord('Q')]:
                break

            # ---- AUTO SAVE ----
            if collecting and res.pose_landmarks:
                now = time.time()

                # 1) Upper-body visibility filter
                visible = sum(
                    1 for i in UPPER_BODY
                    if res.pose_landmarks.landmark[i].visibility > 0.25
                )
                if visible < len(UPPER_BODY) * 0.5:
                    continue

                # 2) Duplicate filter
                vec = []
                for lm in res.pose_landmarks.landmark:
                    vec += [lm.x, lm.y, lm.z]

                if last_vec is not None:
                    diff = sum(abs(a - b) for a, b in zip(vec, last_vec))
                    if diff < 0.005:
                        continue

                # 3) Time filter 300ms
                if now - last_save_time >= 0.3:
                    save_row(current_label, res.pose_landmarks)
                    last_save_time = now
                    last_vec = vec

                    print(f"[Saved] {LABELS[current_label]} — Tổng: {sample_counts[current_label]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
