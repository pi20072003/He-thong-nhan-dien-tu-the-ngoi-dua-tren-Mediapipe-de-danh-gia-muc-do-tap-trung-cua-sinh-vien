"""
Pose Capture (TF 2.15.0 + MediaPipe 0.10.21 + protobuf 4.25.3)

Phím tắt:
  0-2 : đổi nhãn hiện tại
  s   : lưu 1 mẫu (1 frame) vào CSV
  ESC : thoát

LABELS :
    0: "ngoi thang",
    1: "guc dau",
    2: "nga nguoi",
    3: "quay trai",
    4: "quay phai",
    5: "chong tay",

"""

import os, csv, cv2
import mediapipe as mp

# ----- Đường dẫn -----
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_FILE = os.path.join(DATA_DIR, "pose_dataset.csv")

# ----- Bảng nhãn -----
LABELS = {
    0: "ngoi thang",
    1: "guc dau",
    2: "nga nguoi",
    3: "quay trai",
    4: "quay phai",
    5: "chong tay",
}

# ----- Tạo header CSV nếu chưa có -----
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        w = csv.writer(f)
        header = ["label"]
        for i in range(33):  # 33 landmarks
            header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
        w.writerow(header)

# ----- Khởi tạo MediaPipe Pose -----
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def save_row(label_id, pose_landmarks):
    row = [label_id]
    for lm in pose_landmarks.landmark:
        row += [lm.x, lm.y, lm.z, lm.visibility]
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam.")

    current_label = 0
    print("Nhấn phím số (0-5) để chọn nhãn; nhấn 's' để lưu 1 mẫu; ESC để thoát.")

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
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # HUD
            cv2.putText(frame, f"Label [{current_label}]: {LABELS[current_label]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, "Press number 0-5 to change label, 's' to save, ESC to quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            cv2.imshow("Pose Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            # Đổi nhãn
            if key in [ord(str(i)) for i in range(10)]:
                current_label = int(chr(key))

            # Lưu 1 mẫu
            elif key == ord('s'):
                if res.pose_landmarks:
                    save_row(current_label, res.pose_landmarks)
                    print(f"Saved 1 sample to {CSV_FILE} as label {current_label} ({LABELS[current_label]})")
                else:
                    print("Không thấy pose_landmarks — không lưu.")

            elif key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
