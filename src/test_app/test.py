#Giao diện giám sát tư thế thông minh
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import queue
import time
import json
import os
from datetime import datetime   
import pickle
from collections import deque
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

try:
    import mediapipe as mp
    from tensorflow.keras.models import load_model
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# Kết nối serial
try:
    import serial
    ser = serial.Serial('COM3', 115200, timeout=1)
    time.sleep(2)
    SERIAL_AVAILABLE = True
except:
    print("Không thể kết nối serial")
    SERIAL_AVAILABLE = False

class PostureMonitoringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Giám Sát Tư Thế")

        # 2 BIẾN LỊCH SỬ CHÍNH
        self.raw_history = []  # Lưu mỗi giây: (timestamp, posture, confidence)
        self.history = []      # Lưu sau xử lý: (timestamp, posture)

        self.focus_scores = {
                "ngoi thang": 100,  "guc dau": 40,      "nga nguoi": 60,
                "quay trai": 70,    "quay phai": 70,    "chong tay": 80,
                "khong thay nguoi": 0,                  "unknown": 0 }
        
        self.setup_window_size()
        self.root.configure(bg='#f8fafc')
        
        # Đường dẫn các tệp
        self.model_path = "models/posture_ann.h5"
        self.scaler_path = "models/scaler.pkl"
        self.labels_path = "models/labels.json"
        
        # Các biến trạng thái
        self.is_monitoring = False
        self.current_posture = "unknown"
        self.confidence = 0.0
        self.session_start_time = None
        self.session_time = 0
        self.good_posture_time = 0
        self.alerts_count = 0
        self.log_file_path = None
        self.longest_good_streak = 0
        self.current_good_streak = 0

        self.current_led_state = 'O'
        self.last_posture_change_time = 0

        # Google Drive
        self.drive = None
        self.setup_google_drive()
        self.bad_posture_log = {}
        self.last_uploaded_second = 0   # giây cuối đã upload

        self.active_posture = None          # tư thế đang theo dõi
        self.active_start_time = None       # thời điểm bắt đầu theo dõi
        self.last_logged_step = 0           # số bước 10s đã ghi (1 = 10s, 2 = 20s...)

        self.bad_posture_log = {}
        self.no_person_last_step = 0

        self.drive_queue = queue.Queue()
        self.drive_thread = threading.Thread(target=self._drive_worker, daemon=True)
        self.drive_thread.start()

        # Tính thời gian duy trì tư thế
        self.good_stable_time = 0
        self.bad_time = 0
        self.posture_start_time = None
        self.last_stable_posture = None

        self.last_alert_time = None
        self.alert_cooldown_seconds = 1
        
        self.posture_buffer = deque(maxlen=10)
        self.recent_alerts = deque(maxlen=10)
        
        # Các thành phần mô hình
        self.model = None
        self.scaler = None
        self.labels = {"0": "ngoi thang", "1": "guc dau", "2": "nga nguoi",
                       "3": "quay trai", "4": "quay phai", "5": "chong tay",
                       "6": "khong thay nguoi"}
        
        # Camera
        self.cap = None
        
        # MediaPipe
        if MODULES_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
        else:
            self.mp_pose = None
            self.mp_draw = None
        
        self.setup_gui()
        self.load_model()
        self.start_timer()
        
        self.alert_labels = []
    
    def send_led_command(self, cmd):
        """Gửi lệnh LED qua UART"""
        if not SERIAL_AVAILABLE:
            return
        
        if cmd != self.current_led_state:
            try:
                ser.write(cmd.encode())
                self.current_led_state = cmd
                print(f"Đã gửi LED: {cmd}")
            except Exception as e:
                print(f"Lỗi gửi UART: {e}")

    def setup_window_size(self):
        """Tự động điều chỉnh kích thước cửa sổ theo màn hình"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        max_width = min(int(screen_width * 0.85), 1300)
        max_height = min(int(screen_height * 0.8), 850)
        
        width = max(max_width, 1200)
        height = max(max_height, 750)
        
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(1200, 750)
        self.root.maxsize(1200, 750)
        self.root.resizable(False, False)
        
        self.window_width = width
        self.window_height = height
    
    def setup_gui(self):
        main_container = tk.Frame(self.root, bg='#f8fafc')
        main_container.pack(fill='both', expand=True)

        self.create_header(main_container)

        content_frame = tk.Frame(main_container, bg='#f8fafc')
        content_frame.pack(fill='both', expand=True)

        self.create_main_content(content_frame)
    
    def create_header(self, parent):
        """Tạo header compact với trạng thái làm việc"""
        header_frame = tk.Frame(parent, bg='#f8fafc')
        header_frame.pack(fill='x', padx=20, pady=0)
        
        title_label = tk.Label(header_frame, text="Hệ thống giám sát ", 
                              font=('Segoe UI', 18, 'bold'), 
                              fg='#1f2937', bg='#f8fafc')
        title_label.pack(side='left')
        
        subtitle_label = tk.Label(header_frame, text="Giám sát tư thế thông minh", 
                                 font=('Segoe UI', 10), 
                                 fg='#6b7280', bg='#f8fafc')
        subtitle_label.pack(side='left', padx=(15, 0))
    
    def create_main_content(self, parent):
        """Tạo nội dung chính"""
        content_frame = tk.Frame(parent, bg='#f8fafc')
        content_frame.pack(fill='both', expand=True, padx=20, pady=0)
        
        # Left column (camera)
        left_column = tk.Frame(content_frame, bg='#f8fafc')
        left_column.pack(side='left', fill='y', padx=(0, 15))
        self.setup_camera_section(left_column)
        
        # Right column
        right_column = tk.Frame(content_frame, bg='#f8fafc')
        right_column.pack(side='left', fill='y', expand=True)

        left_right_split = tk.Frame(right_column, bg='#f8fafc')
        left_right_split.pack(fill='both', expand=True)

        stats_settings_col = tk.Frame(left_right_split, bg='#f8fafc')
        stats_settings_col.pack(side='left', fill='y', padx=(0, 15))

        stats_frame = self.create_card(stats_settings_col, "Thống kê phiên làm việc")
        stats_frame.pack(fill='x', pady=(0, 15))
        stats_container = tk.Frame(stats_frame, bg='white')
        stats_container.pack(fill='x', padx=15, pady=(0, 15))

        time_frame = self.create_stat_box(stats_container, "00:00:00", "Thời gian làm việc", '#3b82f6', '#dbeafe')
        time_frame.pack(fill='x', pady=(0, 8))
        self.session_time_label = time_frame.children['!label']

        good_frame = self.create_stat_box(stats_container, "0%", "Tư thế tốt", '#10b981', '#d1fae5')
        good_frame.pack(fill='x', pady=(0, 8))
        self.good_posture_label = good_frame.children['!label']

        alert_frame = self.create_stat_box(stats_container, "0", "Cảnh báo", '#ef4444', '#fee2e2')
        alert_frame.pack(fill='x', pady=(0, 8))
        self.alerts_count_label = alert_frame.children['!label']

        self.setup_settings_panel(stats_settings_col)

        alerts_col = tk.Frame(left_right_split, bg='#f8fafc')
        alerts_col.pack(side='left', fill='y')
        self.setup_alerts_panel(alerts_col)
        
        # Khung biểu đồ phân tích
        chart_card = tk.Frame(alerts_col, bg="white", relief="solid", bd=1)
        chart_card.pack(fill="x", pady=(0, 15))

        chart_header = tk.Frame(chart_card, bg="white")
        chart_header.pack(fill="x", padx=15, pady=(10, 5))

        tk.Label(
            chart_header, text="Phân tích file log",
            font=("Segoe UI", 10, "bold"),
            bg="white", fg="#000000",
            anchor="w").pack(side="left")

        self.chart_border = tk.Frame(chart_card, bg="white", height=185, width=330)
        self.chart_border.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        self.chart_border.pack_propagate(False)

        self.analysis_frame = tk.Frame(self.chart_border, bg="white")
        self.analysis_frame.pack(fill="both", expand=True, padx=2, pady=2)
        self.analysis_frame.pack_propagate(False)

        self.analysis_placeholder = tk.Label(
            self.analysis_frame,
            text="Thêm file log để đánh giá",
            font=('Segoe UI', 9, 'italic'),
            bg='white', fg='gray'
        )
        self.analysis_placeholder.pack(pady=20)

    def setup_camera_section(self, parent):
        """Camera section"""
        camera_frame = self.create_card(parent, "Giám sát trực tiếp")
        camera_frame.pack(fill='x', pady=(0, 15))
        
        control_frame = tk.Frame(camera_frame, bg='white')
        control_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        self.start_btn = tk.Button(control_frame, text="Bắt đầu giám sát", 
                                  command=self.toggle_monitoring,
                                  bg='#3b82f6', fg='white', 
                                  font=('Segoe UI', 10, 'bold'),
                                  padx=20, pady=8, relief='flat')

        self.start_btn.pack(side='left', padx=(0, 10))

        help_btn = tk.Button(control_frame, text="Hướng dẫn sử dụng", 
                            command=self.show_help,
                            bg='#6366f1', fg='white', 
                            font=('Segoe UI', 10, 'bold'),
                            padx=20, pady=8, relief='flat')
        help_btn.pack(side='left', padx=(0, 10))

        about_btn = tk.Button(control_frame, text="Thông tin phần mềm", 
                            command=self.show_about,
                            bg='#6b7280', fg='white', 
                            font=('Segoe UI', 10, 'bold'),
                            padx=20, pady=8, relief='flat')
        about_btn.pack(side='left')

        self.status_frame = tk.Frame(control_frame, bg='white')
        self.status_frame.pack(side='right', padx=(15,0))
        self.status_indicator = tk.Label(self.status_frame, text="●", 
                                    font=('Segoe UI', 14), 
                                    fg='#ef4444', bg='white')
        self.status_indicator.pack(side='left')
        self.status_text = tk.Label(self.status_frame, text="Dừng", 
                            font=('Segoe UI', 10, 'bold'), 
                            fg='#374151', bg='white')
        self.status_text.pack(side='left', padx=(5, 0))

        camera_container_outer = tk.Frame(camera_frame, bg='white')
        camera_container_outer.pack(fill='x', padx=15, pady=(0, 15))
        
        base_width = min(int(self.window_width * 0.5), 700)
        base_height = int(base_width * 3 / 4)
        
        self.camera_container = tk.Frame(camera_container_outer, 
                                        bg='#1f2937', 
                                        relief='solid', bd=2,
                                        width=base_width, 
                                        height=base_height)
        self.camera_container.pack(anchor='center')
        self.camera_container.pack_propagate(False)
        
        self.camera_label = tk.Label(self.camera_container, 
                                    text="Camera dừng\nBấm 'Bắt đầu giám sát' để khởi động",
                                    bg='#1f2937', fg='white',
                                    font=('Segoe UI', 12),
                                    anchor='center')
        self.camera_label.place(x=0, y=0, width=base_width, height=base_height)
        
        self.camera_width = base_width
        self.camera_height = base_height
        
        status_frame = tk.Frame(camera_frame, bg='white')
        status_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        posture_frame = tk.Frame(status_frame, bg='#f3f4f6', relief='flat', bd=1)
        posture_frame.pack(side='left', fill='both', expand=True, padx=(0, 8))
        
        tk.Label(posture_frame, text="Tư thế hiện tại", 
                font=('Segoe UI', 9), fg='#6b7280', bg='#f3f4f6').pack(pady=(12, 3))
        
        self.posture_label = tk.Label(posture_frame, text="Chưa xác định", 
                                     font=('Segoe UI', 11, 'bold'),
                                     fg='#374151', bg='#f3f4f6')
        self.posture_label.pack(pady=(0, 12))
        
        confidence_frame = tk.Frame(status_frame, bg='#f3f4f6', relief='flat', bd=1)
        confidence_frame.pack(side='right', fill='both', expand=True, padx=(8, 0))
        
        tk.Label(confidence_frame, text="Độ tin cậy", 
                font=('Segoe UI', 9), fg='#6b7280', bg='#f3f4f6').pack(pady=(12, 3))
        
        self.confidence_label = tk.Label(confidence_frame, text="0.0%", 
                                        font=('Segoe UI', 11, 'bold'),
                                        fg='#374151', bg='#f3f4f6')
        self.confidence_label.pack(pady=(0, 12))

    def setup_alerts_panel(self, parent):
        """Panel cảnh báo"""
        alerts_frame = self.create_card(parent, "Cảnh báo")
        alerts_frame.pack(fill='x', pady=(0, 15))
        
        self.alerts_container = tk.Frame(alerts_frame, bg='white', width=220, height=249)
        self.alerts_container.pack(fill='x', padx=15, pady=(0, 15))
        self.alerts_container.pack_propagate(False)
        
        self.no_alerts_label = tk.Label(self.alerts_container, 
                                       text="Không có cảnh báo mới", 
                                       font=('Segoe UI', 9),
                                       fg='#6b7280', bg='white')
        self.no_alerts_label.pack(pady=15)
 
    def setup_settings_panel(self, parent):
        """Panel cài đặt"""
        settings_frame = self.create_card(parent, "Cài đặt & Điều khiển")
        settings_frame.pack(fill='x')
        
        settings_container = tk.Frame(settings_frame, bg='white')
        settings_container.pack(fill='x', padx=15, pady=(0, 8))
        
        self.sensitivity_var = tk.StringVar(value="Trung bình (≥70%)")
        tk.Label(settings_container, text="Độ nhạy cảnh báo:", 
                font=('Segoe UI', 10), fg='#374151', bg='white').pack(anchor='w', pady=(2, 2))
        self.sensitivity_combo = ttk.Combobox( settings_container, textvariable=self.sensitivity_var,
                                            values=["Thấp (≥50%)", "Trung bình (≥70%)", "Cao (≥90%)"],
                                            state="readonly")
        self.sensitivity_combo.pack(fill='x', pady=(0, 10))
        
        self.export_btn = tk.Button(settings_container, text="Xuất báo cáo", 
                                    command=self.export_report,
                                    bg='#059669', fg='white', font=('Segoe UI', 8, 'bold'),
                                    padx=12, pady=4, relief='flat')
        self.export_btn.pack(fill='x', pady=1)

        self.export_log_btn =  tk.Button(settings_container, text="Xuất file log",
                                            command=self.export_log,
                                            bg="#3b82f6", fg="white", font=('Segoe UI', 8, 'bold'),
                                            padx=12, pady=4, relief='flat')
        self.export_log_btn.pack(fill='x', pady=1)

        self.browse_log_btn = tk.Button(settings_container, text="Browse file log",
                                        command=self.browse_log_file,
                                        bg='#e5e7eb', fg='black', font=('Segoe UI', 8, 'bold'),
                                        padx=12, pady=4, relief='flat')
        self.browse_log_btn.pack(fill='x', pady=1)

        self.camera_btn = tk.Button(settings_container, text="Cài đặt Camera", 
                                    command=self.camera_settings,
                                    bg='#7c3aed', fg='white', font=('Segoe UI', 8, 'bold'),
                                    padx=12, pady=4, relief='flat')
        self.camera_btn.pack(fill='x', pady=1)
    
    def create_card(self, parent, title):
        """Tạo card với header"""
        card = tk.Frame(parent, bg='white', relief='solid', bd=1)
        
        header = tk.Frame(card, bg='white')
        header.pack(fill='x', padx=15, pady=(12, 8))
        
        tk.Label(header, text=title, font=('Segoe UI', 10, 'bold'),
                fg='#1f2937', bg='white').pack(side='left')
        return card
    
    def create_stat_box(self, parent, value, label, color, bg_color):
        """Tạo stat box"""
        box = tk.Frame(parent, bg=bg_color, relief='flat', bd=1)
        
        value_label = tk.Label(box, text=value, font=('Segoe UI', 14, 'bold'),
                              fg=color, bg=bg_color)
        value_label.pack(pady=(10, 3))
        
        tk.Label(box, text=label, font=('Segoe UI', 8),
                fg='#6b7280', bg=bg_color).pack(pady=(0, 10))
        return box
    
    def load_model(self):
        """Tải model đã huấn luyện"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = load_model(self.model_path)
                
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                
                if os.path.exists(self.labels_path):
                    with open(self.labels_path, "r", encoding="utf-8") as f:
                        self.labels = json.load(f)
                
                print("Model loaded successfully!")
            else:
                print("Model files not found. Please train the model first.")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def toggle_monitoring(self):
        """Bật/tắt giám sát"""
        if not self.is_monitoring:
            if not MODULES_AVAILABLE:
                messagebox.showerror("Lỗi", "Các thư viện cần thiết chưa được cài đặt!")
                return
            
            if self.model is None:
                messagebox.showerror("Lỗi", "Model chưa được tải. Vui lòng huấn luyện model trước!")
                return
            
            self.start_monitoring()
        else:
            self.stop_monitoring()
    
    def start_monitoring(self):
        """Bắt đầu giám sát"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở camera!")
                return

            self.session_start_datetime = datetime.now()
            self.session_start_time = time.time()

            # Reset trạng thái
            self.is_monitoring = True
            self.raw_history = []  # Reset raw_history
            self.history = []      # Reset history
            self.posture_buffer.clear()
            
            self.session_time = 0
            self.good_posture_time = 0
            self.alerts_count = 0
            self.last_alert_time = None
            self.bad_posture_start_time = None
            # Reset accumulated bad posture log (avoid carrying over between sessions)
            self.bad_posture_log = {}
            # Helper to track no-person upload steps (10s increments)
            self.no_person_last_step = 0
            # helper to throttle bad-posture console prints (once per second)
            self.last_bad_print_time = 0
            self.good_posture_start_time = None
            self.last_stable_posture = None

            # RESET LED
            self.current_led_state = 'O'
            self.send_led_command('O')
            self.good_stable_time = 0
            self.bad_time = 0

            # Reset alert panel
            self.recent_alerts.clear()
            self.update_alerts_display()

            # Reset chart
            for widget in self.analysis_frame.winfo_children():
                widget.destroy()

            self.analysis_placeholder = tk.Label(
                self.analysis_frame, text="Thêm file log để đánh giá",
                bg='white', fg='gray', 
                font=('Segoe UI', 9, 'italic'))
            self.analysis_placeholder.pack(pady=20)

            # Khóa các nút
            self.sensitivity_combo.config(state="disabled")
            self.export_btn.config(state="disabled")
            self.export_log_btn.config(state="disabled")
            self.browse_log_btn.config(state="disabled")

            # Update UI
            self.start_btn.config(text="Dừng giám sát", bg='#ef4444')
            self.status_indicator.config(fg='#10b981')
            self.status_text.config(text="Đang hoạt động")

            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể bắt đầu giám sát: {e}")

    def stop_monitoring(self):
        """Dừng giám sát"""
        self.session_end_datetime = datetime.now()
        self.monitor_stop_time = time.time()

        self.is_monitoring = False
        if self.cap:
            self.cap.release()

        # Xử lý history trước khi dừng
        self.process_history()

        # Mở lại các control
        self.sensitivity_combo.config(state="readonly")
        self.export_btn.config(state="normal")
        self.export_log_btn.config(state="normal")
        self.browse_log_btn.config(state="normal")

        # TẮT LED
        self.send_led_command('O')
        self.current_led_state = 'O'
        self.good_stable_time = 0
        self.bad_time = 0
        self.bad_posture_start_time = None
        self.good_posture_start_time = None
        self.last_stable_posture = None

        # Update UI
        self.start_btn.config(text="Bắt đầu giám sát", bg='#3b82f6')
        self.status_indicator.config(fg='#ef4444')
        self.status_text.config(text="Dừng")
        self.camera_label.config(image="", text="Camera dừng\nBấm 'Bắt đầu giám sát' để khởi động")
        self.posture_label.config(text="Chưa xác định")
        self.confidence_label.config(text="0.0%")
    
    def camera_loop(self):
        """Vòng lặp xử lý camera"""
        if not self.mp_pose:
            return

        last_logged_time = -1

        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.8,
        ) as pose:

            while self.is_monitoring and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                current_session_time = self.session_time

                # KHÔNG THẤY NGƯỜI
                if not results.pose_landmarks:
                    self.send_led_command('W')
                    self.good_posture_start_time = None
                    self.bad_posture_start_time = None
                    
                    self.current_posture = "khong thay nguoi"
                    self.confidence = 0.0
                    self.posture_buffer.append("khong thay nguoi")

                    # Lưu vào raw_history mỗi giây
                    if current_session_time != last_logged_time:
                        self.raw_history.append((current_session_time, self.current_posture, self.confidence))
                        print(f"[{current_session_time}s] Không thấy người")
                        last_logged_time = current_session_time
                    
                    self.check_alerts("khong thay nguoi")

                    self.root.after(0, self.update_posture_display)

                else:
                    # CÓ NGƯỜI
                    self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
                    features = self.extract_features(results.pose_landmarks)
                    if features is not None:
                        posture, confidence = self.predict_posture(features)
                        self.current_posture = posture
                        self.confidence = confidence
                        
                        self.posture_buffer.append(posture)
                        
                        # Lưu vào raw_history
                        if current_session_time != last_logged_time:
                            self.raw_history.append((current_session_time, posture, confidence))
                            print(f"[{current_session_time}s] {posture} ({confidence:.1%})")
                            last_logged_time = current_session_time

                        if len(self.posture_buffer) >= 5:
                            try:
                                stable_posture = max(set(self.posture_buffer), key=self.posture_buffer.count)
                                count = self.posture_buffer.count(stable_posture)

                                STABILITY_THRESHOLD = 2

                                sensitivity_text = self.sensitivity_var.get()
                                if "50" in sensitivity_text:
                                    threshold = 0.50
                                elif "90" in sensitivity_text:
                                    threshold = 0.90
                                else:
                                    threshold = 0.70

                                if count >= STABILITY_THRESHOLD and (self.confidence >= threshold or stable_posture == "khong thay nguoi"):
                                    self.current_posture = stable_posture
                                    
                                    # ĐIỀU KHIỂN LED
                                    current_time = time.time()

                                    is_good_posture = (stable_posture == "ngoi thang")
                                    is_no_person = (stable_posture == "khong thay nguoi")
                                    is_bad_posture = not is_good_posture and not is_no_person

                                    # TƯ THẾ TỐT
                                    if is_good_posture:
                                        if self.good_posture_start_time is None:
                                            self.good_posture_start_time = current_time
                                            #-print("Bắt đầu đếm tư thế TỐT")

                                        self.bad_posture_start_time = None

                                        good_time = current_time - self.good_posture_start_time

                                        if good_time >= 10:
                                            self.send_led_command('B')
                                        else:
                                            self.send_led_command('O')

                                    # TƯ THẾ XẤU
                                    elif is_bad_posture:
                                        current_time = time.time()

                                        if self.bad_posture_start_time is None:
                                            self.bad_posture_start_time = current_time
                                        else:
                                            bad_time = current_time - self.bad_posture_start_time
                                            #-print("Bad time:", bad_time)

                                            if bad_time >= 10:   # NGƯỠNG 10s
                                                self.update_bad_posture_log(stable_posture, bad_time)
                                                self.bad_posture_start_time = time.time()  # reset mốc đếm

                                        if self.bad_posture_start_time is None:
                                            self.bad_posture_start_time = current_time
                                            #-print("Bắt đầu đếm tư thế XẤU")

                                        self.good_posture_start_time = None

                                        bad_time = current_time - self.bad_posture_start_time
                                        # Throttle console output to once per second and print in requested format
                                        now_ts = time.time()
                                        if now_ts - getattr(self, 'last_bad_print_time', 0) >= 1:
                                            # If raw_history already printed this second, skip duplicate print
                                            if current_session_time != last_logged_time:
                                                print(f"[{current_session_time}s] {stable_posture} ({self.confidence*100:.1f}%)")
                                                self.last_bad_print_time = now_ts
                                        
                                        # ======== GỬI CẢNH BÁO LÊN GOOGLE DRIVE ========
                                        BAD_THRESHOLD = 10  # số giây cảnh báo
                                        if bad_time >= BAD_THRESHOLD:
                                            if not hasattr(self, "last_alert_time"):
                                                self.last_alert_time = None
                                            now = datetime.now()
                                            # tránh gửi liên tục — 60 giây báo lại 1 lần
                                            if (self.last_alert_time is None) or ((now - self.last_alert_time).total_seconds() > 60):
                                                self.send_drive_alert(stable_posture, bad_time)
                                                self.last_alert_time = now

                                        # ======== GỬI TÍN HIỆU LÊN STM32 ========
                                        if bad_time >= 60:
                                            self.send_led_command('R')
                                        elif bad_time >= 30:
                                            self.send_led_command('Y')
                                        else:
                                            self.send_led_command('O')

                                    self.last_stable_posture = stable_posture
                                    self.check_alerts(stable_posture)

                            except (ValueError, TypeError, IndexError) as e:
                                print(f"Lỗi xử lý buffer: {e}")

                        self.root.after(0, self.update_posture_display)

                # Xử lý hiển thị camera
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    display_width = self.camera_width - 10
                    display_height = self.camera_height - 10
                    original_width, original_height = frame_pil.size
                    scale_w = display_width / original_width
                    scale_h = display_height / original_height
                    scale = max(scale_w, scale_h)
                    new_width = int(original_width * scale)
                    new_height = int(original_height * scale)
                    frame_pil = frame_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    left = (new_width - display_width) // 2
                    top = (new_height - display_height) // 2
                    right = left + display_width
                    bottom = top + display_height
                    frame_pil = frame_pil.crop((left, top, right, bottom))
                    frame_tk = ImageTk.PhotoImage(frame_pil)
                    self.root.after(0, lambda img=frame_tk: self.update_camera_display(img))
                except Exception as e:
                    print(f"Lỗi xử lý hình ảnh: {e}")
                time.sleep(0.03)

    def process_history(self):
        """Xử lý raw_history thành history theo logic:
        Bước 1: Kiểm tra ngưỡng (không đạt → unknown)
        Bước 2: Lọc nhiễu (< 3s → thay = tư thế trước, ≥ 3s → giữ nguyên)
        """
        if not self.raw_history:
            return

        sensitivity_text = self.sensitivity_var.get()
        if "50" in sensitivity_text:
            threshold = 0.50
        elif "90" in sensitivity_text:
            threshold = 0.90
        else:
            threshold = 0.70

        # BƯỚC 1: Kiểm tra ngưỡng
        temp_list = []
        for timestamp, posture, confidence in self.raw_history:
            if confidence >= threshold or posture == "khong thay nguoi":
                temp_list.append((timestamp, posture))
            else:
                temp_list.append((timestamp, "unknown"))
        
        if not temp_list:
            return

        # BƯỚC 2: Lọc nhiễu
        # Tìm các đoạn tư thế liên tục
        segments = []
        current_posture = temp_list[0][1]
        start_time = temp_list[0][0]
        
        for i in range(1, len(temp_list)):
            timestamp, posture = temp_list[i]
            if posture != current_posture:
                # Kết thúc đoạn hiện tại
                segments.append({
                    'start': start_time,
                    'end': temp_list[i-1][0],
                    'posture': current_posture,
                    'duration': temp_list[i-1][0] - start_time + 1
                })
                # Bắt đầu đoạn mới
                current_posture = posture
                start_time = timestamp
        
        # Thêm đoạn cuối cùng
        segments.append({
            'start': start_time,
            'end': temp_list[-1][0],
            'posture': current_posture,
            'duration': temp_list[-1][0] - start_time + 1
        })

        # Xử lý từng đoạn: < 3s → thay = tư thế trước, ≥ 3s → giữ
        processed_segments = []
        for i, seg in enumerate(segments):
            if seg['duration'] >= 3:
                # Nếu trùng tư thế với đoạn trước đó (do đoạn giữa bị coi là nhiễu và gộp vào) -> Gộp tiếp
                if processed_segments and processed_segments[-1]['posture'] == seg['posture']:
                    processed_segments[-1]['end'] = seg['end']
                    processed_segments[-1]['duration'] = processed_segments[-1]['end'] - processed_segments[-1]['start'] + 1
                else:
                    processed_segments.append(seg)
            else:
                # Nhiễu → thay = tư thế trước
                if processed_segments:
                    processed_segments[-1]['end'] = seg['end']
                    processed_segments[-1]['duration'] = processed_segments[-1]['end'] - processed_segments[-1]['start'] + 1

        # Chuyển thành history
        self.history = []
        for seg in processed_segments:
            self.history.append((seg['start'], seg['posture']))
        
        print(f"Đã xử lý history: {len(self.raw_history)} dòng raw → {len(self.history)} đoạn")

    def extract_features(self, pose_landmarks):
        """Trích xuất đặc trưng từ pose landmarks"""
        if not pose_landmarks:
            return None
        features = []
        for landmark in pose_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(features, dtype=np.float32)
    
    def predict_posture(self, features):
        """Dự đoán tư thế sử dụng model đã huấn luyện"""
        if self.model is None or self.scaler is None:
            return "unknown", 0.0
        try:
            X = self.scaler.transform(features.reshape(1, -1))
            probs = self.model.predict(X, verbose=0)[0]
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            
            posture_name = self.labels.get(str(class_id), "unknown")
            return posture_name, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return "unknown", 0.0
    
    def update_camera_display(self, image):
        """Cập nhật hiển thị camera"""
        self.camera_label.config(image=image, text="")
        self.camera_label.image = image 
    
    def update_posture_display(self):
        """Cập nhật thông tin tư thế"""
        posture_names = {
            "ngoi thang": "Ngồi thẳng",
            "guc dau": "Gục đầu", 
            "nga nguoi": "Ngả người",
            "quay trai": "Quay trái",
            "quay phai": "Quay phải",
            "chong tay": "Chống tay",
            "khong thay nguoi": "Không thấy người",
            "unknown": "Chưa xác định"
        }
        display_name = posture_names.get(self.current_posture, self.current_posture)
        self.posture_label.config(text=display_name)
        
        confidence_pct = self.confidence * 100
        self.confidence_label.config(text=f"{confidence_pct:.1f}%")
        
        if self.current_posture == "ngoi thang":
            self.posture_label.config(fg='#10b981')
        elif self.current_posture == "khong thay nguoi":
            self.posture_label.config(fg='#6b7280')
        else:
            self.posture_label.config(fg='#ef4444')
    
    def check_alerts(self, posture):
        """Kiểm tra và tạo cảnh báo"""
        current_time = datetime.now()

        # Xử lý riêng cho không thấy người
        if posture == "khong thay nguoi":
            # Start timer for no-person if not present
            if not hasattr(self, 'no_person_start_time'):
                self.no_person_start_time = current_time
                self.no_person_last_step = 0
                return

            duration = (current_time - self.no_person_start_time).total_seconds()

            # Send an upload to Drive every 10s (same logic as other bad postures)
            try:
                step = int(duration // 10)
            except Exception:
                step = 0

            if step >= 1 and step > getattr(self, 'no_person_last_step', 0):
                # delta time since last step
                delta = duration - (getattr(self, 'no_person_last_step', 0) * 10)
                if delta <= 0:
                    delta = 10
                self.no_person_last_step = step
                # record to bad_posture_log and enqueue upload
                try:
                    self.update_bad_posture_log('khong thay nguoi', delta)
                except Exception as e:
                    print('Failed to log no-person to Drive:', e)

            # For UI alerts, keep the existing 60s rule
            if duration >= 60:
                if not self.last_alert_time or (current_time - self.last_alert_time).total_seconds() >= 60:
                    self.last_alert_time = current_time
                    alert_message = "Không phát hiện người \ntrong 1 phút"
                    self.recent_alerts.appendleft((current_time, alert_message))
                    self.alerts_count += 1
                    self.root.after(0, self.update_alerts_display)

            return
        else:
            if hasattr(self, 'no_person_start_time'):
                del self.no_person_start_time

        sensitivity_text = self.sensitivity_var.get()
        if "50" in sensitivity_text:
            threshold = 0.50
        elif "90" in sensitivity_text:
            threshold = 0.90
        else:
            threshold = 0.70

        if posture != "khong thay nguoi" and self.confidence < threshold:
            return

        if self.last_alert_time and (current_time - self.last_alert_time).total_seconds() < self.alert_cooldown_seconds:
            return

        self.last_alert_time = current_time

        alert_names = {
                "guc dau": "Tư thế không tốt: Gục đầu",
                "nga nguoi": "Tư thế không tốt: Ngả người",
                "ngoi thang": "Tư thế tốt: Ngồi thẳng",
                "quay trai": "Tư thế không tốt: Quay trái",
                "quay phai": "Tư thế không tốt: Quay phải",
                "chong tay": "Tư thế không tốt: Chống tay",
        }

        alert_message = alert_names.get(posture, f"Tư thế: {posture}")

        self.recent_alerts.appendleft((current_time, alert_message))
    
        if posture != "ngoi thang":
            self.alerts_count += 1
            #-print(f"Đã thêm cảnh báo: {alert_message} (Độ tin cậy: {self.confidence:.1%}, Ngưỡng: {threshold:.0%})")

        self.root.after(0, self.update_alerts_display)
    
    def update_alerts_display(self):
        """Cập nhật panel cảnh báo"""
        max_alerts = 3
        alerts = list(self.recent_alerts)[:max_alerts]

        while len(self.alert_labels) < max_alerts:
            frame = tk.Frame(self.alerts_container, bg='white')
            frame.pack(fill='x', expand=True, padx=3, pady=4)
            icon_label = tk.Label(frame, font=('Segoe UI', 12), bg='white')
            icon_label.pack(side='left', padx=(8, 5), pady=8)
            content_frame = tk.Frame(frame, bg='white')
            content_frame.pack(side='left', fill='both', expand=True, pady=8)
            msg_label = tk.Label(content_frame, font=('Segoe UI', 9, 'bold'), bg='white', anchor='w')
            msg_label.pack(fill='x', expand=True, anchor='w')
            time_label = tk.Label(content_frame, font=('Segoe UI', 8), bg='white', anchor='w')
            time_label.pack(fill='x', anchor='w')
            self.alert_labels.append((frame, icon_label, msg_label, time_label))

        for i in range(max_alerts):
            if i < len(alerts):
                alert_time, message = alerts[i]
                if "Ngồi thẳng" in message:
                    bg_color = '#d1fae5'
                    fg_color = '#059669'
                    icon = "✅"
                else:
                    bg_color = '#fef2f2'
                    fg_color = '#dc2626'
                    icon = "⚠️"
                frame, icon_label, msg_label, time_label = self.alert_labels[i]
                frame.config(bg=bg_color)
                icon_label.config(text=icon, bg=bg_color)
                content_frame = icon_label.master.children[list(icon_label.master.children.keys())[1]]
                content_frame.config(bg=bg_color)
                msg_label.config(text=message, fg=fg_color, bg=bg_color)
                time_label.config(text=alert_time.strftime("%H:%M:%S"), fg='#374151', bg=bg_color)
                frame.pack(fill='x', expand=True, padx=3, pady=4)
            else:
                frame, _, _, _ = self.alert_labels[i]
                frame.pack_forget()

        if not alerts:
            if not hasattr(self, 'no_alerts_label') or not self.no_alerts_label.winfo_ismapped():
                self.no_alerts_label = tk.Label(self.alerts_container, 
                                                text="Không có cảnh báo mới", 
                                                font=('Segoe UI', 9),
                                                fg='#6b7280', bg='white')
                self.no_alerts_label.pack(pady=15)
        else:
            if hasattr(self, 'no_alerts_label') and self.no_alerts_label.winfo_ismapped():
                self.no_alerts_label.pack_forget()
    
    def start_timer(self):
        """Bắt đầu timer phiên làm việc"""
        def update_timer():
            if self.is_monitoring and hasattr(self, "session_start_datetime"):
                delta = datetime.now() - self.session_start_datetime
                self.session_time = int(delta.total_seconds())

                if self.current_posture == "ngoi thang":
                    self.good_posture_time += 1

                self.update_session_stats()

            self.root.after(1000, update_timer)
        update_timer()

    def update_session_stats(self):
        """Cập nhật thống kê phiên làm việc"""
        hours = self.session_time // 3600
        minutes = (self.session_time % 3600) // 60
        seconds = self.session_time % 60
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        self.session_time_label.config(text=time_str)
        
        # Sử dụng biến đếm trực tiếp từ start_timer
        good_percentage = (self.good_posture_time / self.session_time) * 100 if self.session_time > 0 else 0
        self.good_posture_label.config(text=f"{good_percentage:.0f}%")

        self.alerts_count_label.config(text=str(self.alerts_count))

    def export_report(self):
        """Xuất báo cáo (dùng history đã xử lý)"""
        if not self.history:
            messagebox.showwarning("Cảnh báo", "Chưa có dữ liệu để xuất báo cáo.")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write("==== BÁO CÁO GIÁM SÁT TƯ THẾ ====\n")
                    f.write("-" * 50 + "\n")
            
                    if hasattr(self, "session_start_datetime"):
                        f.write(f"Thời gian bắt đầu: {self.session_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    if hasattr(self, "session_end_datetime"):
                        f.write(f"Thời gian kết thúc: {self.session_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    f.write(f"Thời gian xuất báo cáo: {now_str}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Độ nhạy cảnh báo: {self.sensitivity_var.get()}\n")
                    f.write(f"Thời gian làm việc: {self.session_time // 3600:02d}:{(self.session_time % 3600) // 60:02d}:{self.session_time % 60:02d}\n")
                    f.write(f"Số cảnh báo: {self.alerts_count}\n")
            
                    f.write("\n" + "=" * 50 + "\n")
                
                    f.write("\nTHỐNG KÊ TƯ THẾ\n")
                    f.write("-" * 50 + "\n")
            
                    if self.history:
                        posture_stats = {}
                        total_time_in_posture = {}
                
                        for timestamp, posture in self.history:
                            posture_stats[posture] = posture_stats.get(posture, 0) + 1
                
                        for i in range(len(self.history) - 1):
                            current_ts, current_posture = self.history[i]
                            next_ts, _ = self.history[i + 1]
                            duration = next_ts - current_ts
                            total_time_in_posture[current_posture] = total_time_in_posture.get(current_posture, 0) + duration
                
                        if self.history:
                            last_ts, last_posture = self.history[-1]
                            duration = self.session_time - last_ts
                            total_time_in_posture[last_posture] = total_time_in_posture.get(last_posture, 0) + duration
                
                        total_changes = len(self.history)
                        total_session_time = self.session_time
                
                        f.write("THỐNG KÊ THEO SỐ LẦN:\n")
                        if posture_stats:
                            for posture, count in sorted(posture_stats.items(), key=lambda x: x[1], reverse=True):
                                percentage = (count / total_changes) * 100
                                f.write(f"  {posture}: {count} lần ({percentage:.1f}%)\n")
                        else:
                            f.write("  Không có dữ liệu thống kê số lần\n")
                
                        f.write("\nTHỐNG KÊ THEO THỜI GIAN:\n")
                        if total_time_in_posture:
                            for posture, time_spent in sorted(total_time_in_posture.items(), key=lambda x: x[1], reverse=True):
                                percentage = (time_spent / total_session_time) * 100 if total_session_time > 0 else 0
                                minutes = time_spent // 60
                                seconds = time_spent % 60
                                f.write(f"  {posture}: {minutes:02d}:{seconds:02d} ({percentage:.1f}%)\n")
                        else:
                            f.write("  Không có dữ liệu thống kê thời gian\n")
                
                        f.write("=" * 50 + "\n\n")
                        f.write("ĐÁNH GIÁ HIỆU SUẤT\n")
                        f.write("-" * 50 + "\n")
                        
                        # Tính từ dữ liệu đã xử lý (chính xác hơn biến đếm realtime)
                        good_time_processed = total_time_in_posture.get("ngoi thang", 0)
                        good_posture_percentage_ui = (good_time_processed / self.session_time) * 100 if self.session_time > 0 else 0
                    
                        f.write(f"Thời gian tư thế tốt: {good_posture_percentage_ui:.1f}%\n")
                
                        focus_score_total = 0
                        for posture, time_spent in total_time_in_posture.items():
                            score = self.focus_scores.get(posture, 0)
                            focus_score_total += score * time_spent
                        avg_focus_score = focus_score_total / total_session_time if total_session_time > 0 else 0
                        f.write(f"Điểm tập trung trung bình: {avg_focus_score:.1f}%\n")
                        
                    else:
                        f.write("Không có dữ liệu lịch sử để thống kê.\n")
                
                    f.write("=" * 50 + "\n\n")

                    f.write("LỊCH SỬ THAY ĐỔI TƯ THẾ CHI TIẾT (ĐÃ XỬ LÝ)\n")
                    f.write("-" * 50 + "\n")
                    if not self.history:
                        f.write("Không có dữ liệu lịch sử.\n")
                    else:
                        f.write(f"Tổng số mục trong lịch sử: {len(self.history)}\n\n")
                        for timestamp, posture in self.history:
                            hours = timestamp // 3600
                            minutes = (timestamp % 3600) // 60
                            seconds = timestamp % 60
                            f.write(f"[{hours:02d}:{minutes:02d}:{seconds:02d}] {posture}\n")
                    f.write("=" * 50 + "\n")

            messagebox.showinfo("Xuất báo cáo", f"Báo cáo đã được lưu: {file_path}\n")
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xuất báo cáo: {e}")
            print(f"Chi tiết lỗi export: {e}")

    def export_log(self):
        """Xuất file log (dùng raw_history - dữ liệu gốc)"""
        if not self.raw_history:
            messagebox.showwarning("Cảnh báo", "Chưa có dữ liệu để xuất log.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
            title="Lưu file log"
        )
        if not file_path:
            return

        try:
            # Lấy ngưỡng hiện tại
            sensitivity_text = self.sensitivity_var.get()
            if "50" in sensitivity_text:
                threshold = 0.50
            elif "90" in sensitivity_text:
                threshold = 0.90
            else:
                threshold = 0.70

            with open(file_path, "w", encoding="utf-8") as f:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write("===== FILE LOG GIÁM SÁT TƯ THẾ =====\n")
                start_dt = datetime.fromtimestamp(self.session_start_time)
                f.write(f"Thời gian bắt đầu: {start_dt.strftime('%d/%m/%Y %H:%M:%S')}\n")
                end_time = datetime.fromtimestamp(self.monitor_stop_time)
                f.write(f"Thời gian dừng: {end_time.strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"Thời gian làm việc: {self.session_time // 3600:02d}:{(self.session_time % 3600) // 60:02d}:{self.session_time % 60:02d}\n")
                f.write(f"Ngưỡng độ tin cậy: {threshold*100:.0f}%\n")
                f.write(f"Thời gian xuất file log: {now_str}\n")
                f.write("=" * 50 + "\n\n")

                f.write("THỜI GIAN - TƯ THẾ - ĐỘ TIN CẬY\n")
                f.write("-------------------------------------\n")
                
                # Đếm thống kê
                total_lines = len(self.raw_history)
                đạt_ngưỡng = 0
                không_đạt_ngưỡng = 0
                tư_thế_tốt_đạt_ngưỡng = 0
                
                for t, posture, confidence in self.raw_history:
                    timestamp = time.strftime("%H:%M:%S", time.gmtime(t))
                    f.write(f"{timestamp} - {posture} - {confidence*100:.1f}%\n")
                    
                    # Đếm
                    if confidence >= threshold or posture == "khong thay nguoi":
                        đạt_ngưỡng += 1
                        if posture == "ngoi thang":
                            tư_thế_tốt_đạt_ngưỡng += 1
                    else:
                        không_đạt_ngưỡng += 1

                f.write("\n" + "=" * 50 + "\n")
                f.write("THỐNG KÊ PHÂN TÍCH\n")
                f.write("-------------------------------------\n")
                f.write(f"Tổng số dòng nhận dạng: {total_lines}\n")
                f.write(f"Số dòng đạt ngưỡng (≥{threshold*100:.0f}%): {đạt_ngưỡng} ({đạt_ngưỡng/total_lines*100:.1f}%)\n")
                f.write(f"Số dòng không đạt ngưỡng: {không_đạt_ngưỡng} ({không_đạt_ngưỡng/total_lines*100:.1f}%)\n")
                f.write(f"Tư thế tốt đạt ngưỡng: {tư_thế_tốt_đạt_ngưỡng} ({tư_thế_tốt_đạt_ngưỡng/total_lines*100:.1f}%)\n")
                f.write("=" * 50 + "\n")

            messagebox.showinfo("Hoàn tất", f"Đã xuất file log thành công:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu file log:\n{e}")

    def browse_log_file(self):
        """Browse và phân tích file log"""
        file_path = filedialog.askopenfilename(
            title="Chọn file log",
            filetypes=[("Log files", "*.txt")]
        )
        if not file_path:
            return

        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            def is_posture_line(line):
                if " - " not in line:
                    return False
                parts = line.split(" - ")
                if len(parts) != 3:
                    return False
                time_part = parts[0].strip()
                time_parts = time_part.split(":")
                return len(time_parts) == 3 and all(p.isdigit() for p in time_parts)

            posture_lines = [l for l in lines if is_posture_line(l)]

            if not posture_lines:
                messagebox.showwarning(
                    "Chưa có dữ liệu",
                    "File log này không có dữ liệu tư thế để phân tích."
                )
                return

            # File hợp lệ → gán và XÓA PLACEHOLDER CŨ
            self.log_file_path = file_path

            # XÓA PLACEHOLDER (nếu có)
            if hasattr(self, "analysis_placeholder") and self.analysis_placeholder:
                try:
                    self.analysis_placeholder.pack_forget()
                    self.analysis_placeholder.destroy()
                except:
                    pass
                self.analysis_placeholder = None

            # Hiển thị phân tích mới
            self.show_log_analysis()

        except Exception as e:
            messagebox.showerror(
                "Lỗi",
                f"Không đọc được file log:\n{e}"
            )

    def show_log_analysis(self):
        """Hiển thị phân tích file log (dạng text thay vì biểu đồ tròn)"""
        if not self.log_file_path:
            return
        
        if self.is_monitoring:
            return
        
        # XÓA SẠCH TẤT CẢ widgets cũ (bao gồm cả placeholder)
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()
        
        # Reset biến placeholder
        if hasattr(self, "analysis_placeholder"):
            self.analysis_placeholder = None

        try:
            # Đọc file và phân tích
            total_lines = 0
            đạt_ngưỡng = 0
            không_đạt_ngưỡng = 0
            tư_thế_tốt_đạt_ngưỡng = 0
            
            # Lấy ngưỡng từ cài đặt
            sensitivity_text = self.sensitivity_var.get()
            if "50" in sensitivity_text:
                threshold = 0.50
            elif "90" in sensitivity_text:
                threshold = 0.90
            else:
                threshold = 0.70

            with open(self.log_file_path, encoding='utf-8') as f:
                for line in f:
                    if " - " not in line or ":" not in line:
                        continue
                    parts = line.strip().split(" - ")
                    if len(parts) != 3:
                        continue
                    
                    try:
                        _, posture, confidence_str = parts
                        confidence = float(confidence_str.rstrip('%')) / 100
                        
                        total_lines += 1
                        
                        if confidence >= threshold or posture.strip() == "khong thay nguoi":
                            đạt_ngưỡng += 1
                            if posture.strip() == "ngoi thang":
                                tư_thế_tốt_đạt_ngưỡng += 1
                        else:
                            không_đạt_ngưỡng += 1
                    except:
                        continue

            if total_lines == 0:
                # Tạo placeholder mới
                self.analysis_placeholder = tk.Label(
                    self.analysis_frame,
                    text="Không có dữ liệu hợp lệ trong file này",
                    font=('Segoe UI', 9, 'italic'),
                    bg='white', fg='red'
                )
                self.analysis_placeholder.pack(pady=20)
                return

            # Hiển thị thống kê dạng text
            stats_container = tk.Frame(self.analysis_frame, bg='white')
            stats_container.pack(expand=True, pady=10)

            # Tổng dòng
            tk.Label(
                stats_container,
                text=f"Tổng dòng: {total_lines}",
                font=('Segoe UI', 10, 'bold'),
                bg='white', fg='#1f2937'
            ).pack(anchor='w', pady=2)

            # Đạt ngưỡng
            đạt_pct = (đạt_ngưỡng / total_lines * 100) if total_lines > 0 else 0
            tk.Label(
                stats_container,
                text=f"Đạt ngưỡng: {đạt_ngưỡng} ({đạt_pct:.1f}%)",
                font=('Segoe UI', 9),
                bg='white', fg='#10b981'
            ).pack(anchor='w', pady=2)

            # Không đạt ngưỡng
            không_đạt_pct = (không_đạt_ngưỡng / total_lines * 100) if total_lines > 0 else 0
            tk.Label(
                stats_container,
                text=f"Không đạt: {không_đạt_ngưỡng} ({không_đạt_pct:.1f}%)",
                font=('Segoe UI', 9),
                bg='white', fg='#ef4444'
            ).pack(anchor='w', pady=2)

            # Tư thế tốt đạt ngưỡng
            tốt_pct = (tư_thế_tốt_đạt_ngưỡng / total_lines * 100) if total_lines > 0 else 0
            tk.Label(
                stats_container,
                text=f"Tư thế tốt đạt ngưỡng: {tư_thế_tốt_đạt_ngưỡng} ({tốt_pct:.1f}%)",
                font=('Segoe UI', 9),
                bg='white', fg='#3b82f6'
            ).pack(anchor='w', pady=2)

        except Exception as e:
            # Tạo label lỗi mới
            error_label = tk.Label(
                self.analysis_frame,
                text=f"Lỗi phân tích: {str(e)}",
                font=('Segoe UI', 9, 'italic'),
                bg='white', fg='red'
            )
            error_label.pack(pady=20)

    def camera_settings(self):
        """Cài đặt camera"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Cài đặt Camera")
        settings_window.geometry("350x200")
        settings_window.resizable(False, False)
        settings_window.configure(bg='#f8fafc')
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        tk.Label(settings_window, text="ID Camera:", font=('Segoe UI', 10, 'bold'),
                fg='#374151', bg='#f8fafc').pack(pady=(20, 5))
        camera_id_var = tk.StringVar(value="0")
        camera_spinbox = tk.Spinbox(settings_window, from_=0, to=9, width=10, 
                                   textvariable=camera_id_var)
        camera_spinbox.pack(pady=(0, 20))
        
        def test_camera():
            try:
                camera_id = int(camera_id_var.get())
                test_cap = cv2.VideoCapture(camera_id)
                if test_cap.isOpened():
                    messagebox.showinfo("Kiểm tra Camera", f"Camera ID {camera_id} hoạt động bình thường!")
                    test_cap.release()
                else:
                    messagebox.showerror("Lỗi Camera", f"Không thể mở Camera ID {camera_id}!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi kiểm tra camera: {e}")
        
        tk.Button(settings_window, text="Kiểm tra Camera", command=test_camera,
                 bg='#3b82f6', fg='white', font=('Segoe UI', 9, 'bold'),
                 padx=20, pady=8, relief='flat').pack(pady=10)
        
    def setup_google_drive(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(base_dir)
        settings_path = os.path.join(base_dir, "settings.yaml")
        self.gauth = GoogleAuth(settings_path)

        if self.gauth.credentials is None:
            self.gauth.CommandLineAuth()
        elif self.gauth.access_token_expired:
            self.gauth.Refresh()
        else:
            self.gauth.Authorize()

        self.gauth.SaveCredentialsFile("token.json")
        self.drive = GoogleDrive(self.gauth)

        print("Google Drive connected")

    def update_bad_posture_log(self, posture, duration):
        # Nếu chưa có mục này → khởi tạo start_time và tổng thời gian
        if posture not in self.bad_posture_log:
            self.bad_posture_log[posture] = {
                "start_time": datetime.now(),
                "total_duration": duration
            }
        else:
            # Cộng dồn thời gian (không ghi đè) — cập nhật tổng thời gian
            try:
                self.bad_posture_log[posture]["total_duration"] += duration
            except Exception:
                # Fall back: ghi đè nếu có lỗi cấu trúc
                self.bad_posture_log[posture]["total_duration"] = duration

        # Gửi yêu cầu upload (không block)
        self.drive_queue.put(posture)

    def write_posture_log_to_drive(self):
        lines = []
        for posture, info in self.bad_posture_log.items():
            t = info["start_time"].strftime("%Y-%m-%d %H:%M:%S")
            d = info["total_duration"]
            lines.append(f"{t} — {posture} — {d:.1f} giây")

        content = "\n".join(lines)

        file_list = self.drive.ListFile(
            {'q': "title='posture_alerts.txt' and trashed=false"}
        ).GetList()

        if file_list:
            file = file_list[0]
            file.SetContentString(content)
        else:
            file = self.drive.CreateFile({'title': 'posture_alerts.txt'})
            file.SetContentString(content)

        file.Upload()

    def process_posture(self, posture):
        now = time.time()

        if self.active_posture is None: # Lần đầu phát hiện
            self.active_posture = posture
            self.active_start_time = now
            self.last_logged_step = 0
            return
        
        if posture != self.active_posture:  # Phát hiện tư thế khác → reset
            self.active_posture = posture
            self.active_start_time = now
            self.last_logged_step = 0
            return

        duration = now - self.active_start_time # Cùng tư thế → tính thời gian liên tục
        step = int(duration // 10)   # mốc 10s, 20s, 30s...

        # Chỉ update khi vượt mốc mới — truyền delta (không truyền cumulative)
        if step >= 1 and step > self.last_logged_step:
            # delta = phần thời gian mới kể từ mốc trước (ví dụ: lần đầu 10s, lần sau +10s...)
            delta = duration - (self.last_logged_step * 10)
            if delta < 0:
                delta = duration
            self.last_logged_step = step
            self.update_bad_posture_log(self.active_posture, delta)


    def _drive_worker(self):
        while True:
            posture = self.drive_queue.get()
            if posture is None:
                break

            try:
                self.write_posture_log_to_drive()
                print("Drive file updated")
            except Exception as e:
                print("Drive upload failed:", e)
                time.sleep(5)
                self.drive_queue.put(posture)

            self.drive_queue.task_done()
    
    def show_help(self):
        """Hiển thị hướng dẫn"""
        help_text = """Hướng dẫn sử dụng:

1. Bắt đầu giám sát:
   - Bấm nút "Bắt đầu giám sát"
   - Ngồi thẳng và điều chỉnh góc cam hợp lý
   - Hệ thống sẽ tự động phân tích tư thế

2. Theo dõi thống kê:
   - Xem thời gian làm việc
   - Theo dõi % tư thế tốt
   - Nhận cảnh báo khi cần

3. Tư thế được nhận diện:
   - Ngồi thẳng: Tư thế tốt
   - Gục đầu: Cần điều chỉnh
   - Ngả người: Cần điều chỉnh
   - Quay trái/phải: Cần điều chỉnh
   - Chống tay: Cần điều chỉnh

4. File log vs Báo cáo:
   - File log: Hiển thị toàn bộ dữ liệu gốc mỗi giây
   - Báo cáo: Dữ liệu đã xử lý, lọc nhiễu

5. Cách tính điểm tập trung:
   - Ngồi thẳng: 100%      - Gục đầu: 40%
   - Ngả người: 60%        - Quay trái/phải: 70%   
   - Chống tay: 80%        - Không thấy người: 0%

Lưu ý: Ngồi cách camera 60-100cm để đạt độ chính xác tốt nhất."""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Hướng dẫn sử dụng")
        help_window.geometry("500x450")
        help_window.configure(bg='#f8fafc')
        help_window.transient(self.root)
        help_window.grab_set()
        
        text_frame = tk.Frame(help_window, bg='#f8fafc')
        text_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        text_widget = tk.Text(text_frame, wrap='word', font=('Segoe UI', 10),
                             bg='white', fg='#374151')
        scrollbar = ttk.Scrollbar(text_frame, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        text_widget.insert('1.0', help_text)
        text_widget.config(state='disabled')
    
    def show_about(self):
        """Hiển thị thông tin ứng dụng"""
        about_text = """Hệ thống giám sát tư thế thông minh

Hệ thống sử dụng AI để phân tích và cảnh báo 
tư thế ngồi sai.

Công nghệ sử dụng:
• MediaPipe - Nhận diện khung xương người
• TensorFlow - Mô hình AI phân loại tư thế  
• OpenCV - Xử lý hình ảnh camera
• Python & Tkinter - Giao diện người dùng

Tính năng mới:
• Raw History: Lưu dữ liệu gốc mỗi giây
• History: Dữ liệu đã xử lý (lọc nhiễu < 3s)
• File log: Hiển thị toàn bộ dữ liệu + thống kê
• Báo cáo: Dữ liệu đã xử lý, chính xác hơn

Phát triển bởi: 
- Đỗ Quang Huy (pi2007)
- Lê Quang Huy (playmaker)
"""
        messagebox.showinfo("Thông tin Hệ thống giám sát", about_text)
    
def main():
    root = tk.Tk()
    
    def toggle_monitoring_key(event):
        app.toggle_monitoring()
    
    def on_closing():
        if app.is_monitoring:
            app.stop_monitoring()
        if SERIAL_AVAILABLE:
            ser.close()
        root.destroy()
        
    app = PostureMonitoringGUI(root)
    
    root.bind('<F1>', toggle_monitoring_key)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.focus_set()
    
    root.mainloop()

if __name__ == "__main__":
    main()
                    