"""Giao diện giám sát tư thế thông minh"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import json
import os
from datetime import datetime, timedelta
import pickle
from collections import deque
try:
    import mediapipe as mp
    from tensorflow.keras.models import load_model
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

class PostureMonitoringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Giám Sát Tư Thế")
        
        self.setup_window_size()
        self.root.configure(bg='#f8fafc')
        
        self.model_path = "models/posture_ann.h5"
        self.scaler_path = "models/scaler.pkl"
        self.labels_path = "models/labels.json"
        
        self.is_monitoring = False
        self.current_posture = "unknown"
        self.confidence = 0.0
        self.session_start_time = None
        self.session_time = 0
        self.good_posture_time = 0
        self.alerts_count = 0
        
        self.last_alert_time = None
        self.alert_cooldown_seconds = 5  # Chỉ cảnh báo 1 lần mỗi 5 giây
        
        self.posture_history = deque(maxlen=100)
        self.recent_alerts = deque(maxlen=10)
        self.daily_stats = {"good_time": 0, "bad_time": 0, "alerts": 0}
        
        self.model = None
        self.scaler = None
        self.labels = {"0": "ngoi thang", "1": "guc dau", "2": "nga nguoi",
                       "3": "quay trai", "4": "quay phai", "5": "chong tay"}
        
        self.cap = None
        self.current_frame = None
        
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
    
    def setup_window_size(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        max_width = min(int(screen_width * 0.85), 1300)
        max_height = min(int(screen_height * 0.8), 850)
        
        width = max(max_width, 1100)
        height = max(max_height, 700)
        
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(1100, 700)
        self.root.resizable(True, True)
        
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
        
        self.status_frame = tk.Frame(header_frame, bg='#f8fafc')
        self.status_frame.pack(side='right')
        self.status_indicator = tk.Label(self.status_frame, text="●", 
                                        font=('Segoe UI', 14), 
                                        fg='#ef4444', bg='#f8fafc')
        self.status_indicator.pack(side='left')
        self.status_text = tk.Label(self.status_frame, text="Dừng", 
                                   font=('Segoe UI', 10, 'bold'), 
                                   fg='#374151', bg='#f8fafc')
        self.status_text.pack(side='left', padx=(5, 0))
    
    def create_main_content(self, parent):
        """Tạo nội dung chính với camera CỐ ĐỊNH và bố cục mới"""
        content_frame = tk.Frame(parent, bg='#f8fafc')
        content_frame.pack(fill='both', expand=True, padx=20, pady=0)

        left_column = tk.Frame(content_frame, bg='#f8fafc')
        left_column.pack(side='left', fill='y', padx=(0, 15))
        self.setup_camera_section(left_column)
        
        right_column = tk.Frame(content_frame, bg='#f8fafc')
        right_column.pack(side='left', fill='y', padx=(0, 15))
        
        self.setup_alerts_panel(right_column)
        
        stats_frame = self.create_card(right_column, "Thống kê phiên làm việc")
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
        
        right_side_column = tk.Frame(content_frame, bg='#f8fafc')
        right_side_column.pack(side='left', fill='y', padx=(0, 15))
        
        self.setup_quick_stats(right_side_column)
        self.setup_recent_history(right_side_column)
        self.setup_settings_panel(right_side_column)
    
    def setup_camera_section(self, parent):
        camera_frame = self.create_card(parent, "Giám sát trực tiếp")
        camera_frame.pack(fill='x', pady=(0, 15))
        
        control_frame = tk.Frame(camera_frame, bg='white')
        control_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        self.start_btn = tk.Button(control_frame, text="Bắt đầu giám sát", 
                                  command=self.toggle_monitoring,
                                  bg='#3b82f6', fg='white', 
                                  font=('Segoe UI', 10, 'bold'),
                                  padx=20, pady=8, relief='flat')
        self.start_btn.pack(side='left')
        
        settings_btn = tk.Button(control_frame, text="Cài đặt", 
                               command=self.open_settings,
                               bg='#6b7280', fg='white', 
                               font=('Segoe UI', 9),
                               padx=15, pady=8, relief='flat')
        settings_btn.pack(side='right')
        
        camera_container_outer = tk.Frame(camera_frame, bg='white')
        camera_container_outer.pack(fill='x', padx=15, pady=(0, 15))
    
        base_width = min(int(self.window_width * 0.5), 700)
        base_height = int(base_width * 3 / 4)  # 4:3 ratio
        
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
    
    def setup_session_stats(self, parent):
        """Thống kê phiên làm việc"""
        stats_frame = self.create_card(parent, "Thống kê phiên làm việc")
        stats_frame.pack(fill='x', pady=(0, 15))
        
        stats_container = tk.Frame(stats_frame, bg='white')
        stats_container.pack(fill='x', padx=15, pady=(0, 15))
        
        time_frame = self.create_stat_box(stats_container, "00:00:00", "Thời gian làm việc", 
                                         '#3b82f6', '#dbeafe')
        time_frame.pack(side='left', fill='both', expand=True, padx=(0, 8))
        self.session_time_label = time_frame.children['!label']
        
        good_frame = self.create_stat_box(stats_container, "0%", "Tư thế tốt", 
                                         '#10b981', '#d1fae5')
        good_frame.pack(side='left', fill='both', expand=True, padx=(4, 4))
        self.good_posture_label = good_frame.children['!label']
        
        alert_frame = self.create_stat_box(stats_container, "0", "Cảnh báo", 
                                          '#ef4444', '#fee2e2')
        alert_frame.pack(side='left', fill='both', expand=True, padx=(8, 0))
        self.alerts_count_label = alert_frame.children['!label']
    
    def setup_right_column_scroll(self, parent):
        right_canvas = tk.Canvas(parent, bg='#f8fafc', highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=right_canvas.yview)
        right_scrollable = tk.Frame(right_canvas, bg='#f8fafc')
        
        right_scrollable.bind(
            "<Configure>",
            lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        )
        
        right_canvas.create_window((0, 0), window=right_scrollable, anchor="nw")
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        
        right_canvas.pack(side="left", fill="both", expand=True)
        right_scrollbar.pack(side="right", fill="y")
        
        def _on_right_mousewheel(event):
            right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        right_canvas.bind("<MouseWheel>", _on_right_mousewheel)
        
        self.setup_quick_stats(right_scrollable)
        self.setup_recent_history(right_scrollable)
        self.setup_settings_panel(right_scrollable)
    
    def setup_alerts_panel(self, parent):
        alerts_frame = self.create_card(parent, "Cảnh báo")
        alerts_frame.pack(fill='x', pady=(0, 12))
        
        self.alerts_container = tk.Frame(alerts_frame, bg='white', width=220, height=200)
        self.alerts_container.pack(fill='x', padx=15, pady=(0, 15))
        self.alerts_container.pack_propagate(False)
        
        self.no_alerts_label = tk.Label(self.alerts_container, 
                                       text="Không có cảnh báo mới", 
                                       font=('Segoe UI', 9),
                                       fg='#6b7280', bg='white')
        self.no_alerts_label.pack(pady=15)
    
    def setup_quick_stats(self, parent):
        stats_frame = self.create_card(parent, "Thống kê nhanh")
        stats_frame.pack(fill='x', pady=(0, 12))
        
        stats_container = tk.Frame(stats_frame, bg='white')
        stats_container.pack(fill='x', padx=15, pady=(0, 15))

#?????????????????????????????????????????????? 
        stats_data = [
            ("Thời gian tốt nhất", "2h 45m"),
            ("Cải thiện tuần này", "+12%"),
            ("Mục tiêu hôm nay", "6/8 giờ")
        ]
#??????????????????????????????????????????????
        
        for label, value in stats_data:
            row = tk.Frame(stats_container, bg='white')
            row.pack(fill='x', pady=3)
            
            tk.Label(row, text=label, font=('Segoe UI', 9),
                    fg='#6b7280', bg='white').pack(side='left')
            
            color = '#10b981' if '+' in value else '#374151'
            tk.Label(row, text=value, font=('Segoe UI', 9, 'bold'),
                    fg=color, bg='white').pack(side='right')
    
    def setup_recent_history(self, parent):
        history_frame = self.create_card(parent, "Lịch sử gần đây")
        history_frame.pack(fill='x', pady=(0, 12))
        
        history_container = tk.Frame(history_frame, bg='white')
        history_container.pack(fill='x', padx=15, pady=(0, 15))
        
#??????????????????????????????????????????????
        history_data = [
            ("09:00", "sit_straight", "15m"),
            ("09:15", "lean_forward", "8m"),
            ("09:23", "sit_straight", "12m"),
            ("09:35", "head_forward", "5m")
        ]
#??????????????????????????????????????????????

        for time_str, posture, duration in history_data:
            row = tk.Frame(history_container, bg='white')
            row.pack(fill='x', pady=2)
            
            icon = "✅" if posture == "sit_straight" else "❌"
            tk.Label(row, text=icon, font=('Segoe UI', 9),
                    bg='white').pack(side='left')
            
            tk.Label(row, text=time_str, font=('Segoe UI', 8),
                    fg='#6b7280', bg='white').pack(side='left', padx=(8, 0))
            
            tk.Label(row, text=duration, font=('Segoe UI', 8, 'bold'),
                    fg='#374151', bg='white').pack(side='right')
    
    def setup_settings_panel(self, parent):
        settings_frame = self.create_card(parent, "Cài đặt & Điều khiển")
        settings_frame.pack(fill='x')
        
        settings_container = tk.Frame(settings_frame, bg='white')
        settings_container.pack(fill='x', padx=15, pady=(0, 15))
        
        buttons = [
            ("Chế độ toàn màn hình", self.toggle_fullscreen, '#6366f1'),
            ("Xuất báo cáo", self.export_report, '#059669'),
            ("Cài đặt Camera", self.camera_settings, '#7c3aed'),
            ("Hướng dẫn sử dụng", self.show_help, '#0891b2'),
            ("Thông tin phần mềm", self.show_about, '#6b7280')
        ]
        
        for text, command, color in buttons:
            btn = tk.Button(settings_container, text=text, command=command,
                           bg=color, fg='white', font=('Segoe UI', 8),
                           padx=12, pady=4, relief='flat')
            btn.pack(fill='x', pady=1)
    
    def create_card(self, parent, title):
        card = tk.Frame(parent, bg='white', relief='solid', bd=1)
        
        header = tk.Frame(card, bg='white')
        header.pack(fill='x', padx=15, pady=(12, 8))
        
        tk.Label(header, text=title, font=('Segoe UI', 10, 'bold'),
                fg='#1f2937', bg='white').pack(side='left')
        
        return card
    
    def create_stat_box(self, parent, value, label, color, bg_color):
        box = tk.Frame(parent, bg=bg_color, relief='flat', bd=1)
        
        value_label = tk.Label(box, text=value, font=('Segoe UI', 14, 'bold'),
                              fg=color, bg=bg_color)
        value_label.pack(pady=(10, 3))
        
        tk.Label(box, text=label, font=('Segoe UI', 8),
                fg='#6b7280', bg=bg_color).pack(pady=(0, 10))
        return box
    
    def load_model(self):
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
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở camera!")
                return

            self.is_monitoring = True
            self.session_start_time = time.time()
            self.session_time = 0
            self.good_posture_time = 0
            self.alerts_count = 0
            self.last_alert_time = None # Reset thời gian cảnh báo
            
            self.start_btn.config(text="Dừng giám sát", bg='#ef4444')
            self.status_indicator.config(fg='#10b981')
            self.status_text.config(text="Đang hoạt động")
            
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể bắt đầu giám sát: {e}")
    
    def stop_monitoring(self):
        self.is_monitoring = False
        
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(text="Bắt đầu giám sát", bg='#3b82f6')
        self.status_indicator.config(fg='#ef4444')
        self.status_text.config(text="Dừng")
        self.camera_label.config(image="", text="Camera dừng\nBấm 'Bắt đầu giám sát' để khởi động")
        self.posture_label.config(text="Chưa xác định")
        self.confidence_label.config(text="0.0%")
    
    def camera_loop(self):
        if not self.mp_pose:
            return

        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while self.is_monitoring and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                '''
                if results.pose_landmarks:
                    # Lấy các tọa độ landmark
                    h, w, _ = frame.shape
                    x_coords = [lm.x * w for lm in results.pose_landmarks.landmark]
                    y_coords = [lm.y * h for lm in results.pose_landmarks.landmark]

                    # Lấy bounding box quanh cơ thể
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    # Thêm padding cho đẹp
                    padding = 50
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)

                    # Cắt khung hình chỉ còn người
                    frame = frame[y_min:y_max, x_min:x_max]
                '''

                if results.pose_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
                    features = self.extract_features(results.pose_landmarks)
                    if features is not None:
                        posture, confidence = self.predict_posture(features)
                        self.current_posture = posture
                        self.confidence = confidence
                        
                        self.root.after(0, self.update_posture_display)
                        
                        self.check_alerts(posture)
                
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
                
                time.sleep(0.03)  # ~30 FPS
    
    def extract_features(self, pose_landmarks):
        if not pose_landmarks:
            return None
        
        features = []
        for landmark in pose_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        return np.array(features, dtype=np.float32)
    
    def predict_posture(self, features):
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
        self.camera_label.config(image=image, text="")
        self.camera_label.image = image
    
    def update_posture_display(self):
        """Cập nhật thông tin tư thế
            0: "ngoi thang",
            1: "guc dau",
            2: "nga nguoi",
            3: "quay trai",
            4: "quay phai",
            5: "chong tay",
        """
        posture_names = {
            "ngoi thang": "Ngồi thẳng",
            "guc dau": "Cúi đầu", 
            "nga nguoi": "Ngả người",
            "quay trai": "Quay trái",
            "quay phai": "Quay phải",
            "chong tay": "Chống tay",
            "unknown": "Chưa xác định"
        }
        
        display_name = posture_names.get(self.current_posture, self.current_posture)
        self.posture_label.config(text=display_name)
        
        confidence_pct = self.confidence * 100
        self.confidence_label.config(text=f"{confidence_pct:.1f}%")
        
        if self.current_posture == "ngoi thang":
            self.posture_label.config(fg='#10b981')  # Xanh lá
        else:
            self.posture_label.config(fg='#ef4444')  # Đỏ
    
    def check_alerts(self, posture):
        current_time = datetime.now()
        if self.last_alert_time and \
           (current_time - self.last_alert_time).total_seconds() < self.alert_cooldown_seconds:
            return

        self.last_alert_time = current_time  # Cập nhật thời gian cảnh báo cuối

        alert_names = {
            "guc dau": "Tư thế không tốt: Cúi đầu",
            "nga nguoi": "Tư thế không tốt: Ngả người",
            "ngoi thang": "Tư thế tốt: Ngồi thẳng",
            "quay trai": "Tư thế không tốt: Quay trái",
            "quay phai": "Tư thế không tốt: Quay phải",
            "chong tay": "Tư thế không tốt: Chống tay"
        }

        alert_message = alert_names.get(posture, f"Tư thế: {posture}")

        self.recent_alerts.appendleft((current_time, alert_message))
        if posture != "ngoi thang":
            self.alerts_count += 1  # Chỉ tăng số cảnh báo với tư thế xấu

        self.root.after(0, self.update_alerts_display)
    
    def update_alerts_display(self):
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
        def update_timer():
            if self.is_monitoring and self.session_start_time:
                self.session_time = int(time.time() - self.session_start_time)
                
                if self.current_posture == "ngoi thang":
                    self.good_posture_time += 1
                self.update_session_stats()
            self.root.after(1000, update_timer)
        update_timer()
    
    def update_session_stats(self):
        # Định dạng thời gian phiên
        hours = self.session_time // 3600
        minutes = (self.session_time % 3600) // 60
        seconds = self.session_time % 60
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        self.session_time_label.config(text=time_str)
        
        if self.session_time > 0:
            good_percentage = (self.good_posture_time / self.session_time) * 100
            self.good_posture_label.config(text=f"{good_percentage:.0f}%")
        
        self.alerts_count_label.config(text=str(self.alerts_count))
    
    def toggle_fullscreen(self):
        try:
            if self.root.attributes('-fullscreen'):
                self.root.attributes('-fullscreen', False)
                self.root.state('zoomed')
            else:
                self.root.attributes('-fullscreen', True)
        except:
            try:
                self.root.state('zoomed')
            except:
                pass
    
    def export_report(self):
        from tkinter import filedialog
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Báo cáo tư thế\n")
                    f.write("=" * 40 + "\n")
                    f.write(f"Thời gian quay: {self.session_time // 60} phút\n")
                    f.write(f"Số cảnh báo: {self.alerts_count}\n")
                    f.write(f"Tư thế cuối cùng phát hiện: {self.current_posture}\n")
                messagebox.showinfo("Xuất báo cáo", f"Báo cáo đã được lưu: {file_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xuất báo cáo: {e}")
    
    def camera_settings(self):
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
    
    def show_help(self):
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
   - Cúi đầu: Cần điều chỉnh
   - Ngả người: Cần điều chỉnh
   - Quay trái/phải: Cần điều chỉnh
   - Chống tay: Cần điều chỉnh

Lưu ý: Ngồi cách camera 60-100cm để đạt độ chính xác tốt nhất."""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Hướng dẫn sử dụng")
        help_window.geometry("500x400")
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
        about_text = """Hệ thống giám sát  v1.0

Hệ thống giám sát tư thế thông minh sử dụng AI
để phân tích và cảnh báo tư thế ngồi không đúng.

Công nghệ sử dụng:
• MediaPipe - Nhận diện khung xương người
• TensorFlow - Mô hình AI phân loại tư thế  
• OpenCV - Xử lý hình ảnh camera
• Python & Tkinter - Giao diện người dùng

Phát triển bởi: 2 Huy
"""
        messagebox.showinfo("Thông tin Hệ thống giám sát ", about_text)
    
    def open_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Cài đặt")
        settings_window.geometry("400x300")
        settings_window.resizable(False, False)
        settings_window.configure(bg='#f8fafc')
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        title_label = tk.Label(settings_window, text="Cài đặt Hệ thống giám sát ", 
                              font=('Segoe UI', 14, 'bold'), 
                              fg='#1f2937', bg='#f8fafc')
        title_label.pack(pady=(20, 15))
        
        settings_frame = tk.Frame(settings_window, bg='white', relief='solid', bd=1)
        settings_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        sensitivity_frame = tk.Frame(settings_frame, bg='white')
        sensitivity_frame.pack(fill='x', padx=15, pady=15)
        
        tk.Label(sensitivity_frame, text="Độ nhạy cảnh báo:", 
                font=('Segoe UI', 10), fg='#374151', bg='white').pack(anchor='w')
        
        sensitivity_var = tk.StringVar(value="Trung bình")
        sensitivity_combo = ttk.Combobox(sensitivity_frame, textvariable=sensitivity_var,
                                       values=["Thấp", "Trung bình", "Cao"], state="readonly")
        sensitivity_combo.pack(fill='x', pady=(5, 0))
        
        auto_start_frame = tk.Frame(settings_frame, bg='white')
        auto_start_frame.pack(fill='x', padx=15, pady=10)
        
        auto_start_var = tk.BooleanVar()
        auto_start_check = tk.Checkbutton(auto_start_frame, 
                                         text="Tự động bắt đầu giám sát khi khởi động",
                                         variable=auto_start_var, bg='white', 
                                         font=('Segoe UI', 9))
        auto_start_check.pack(anchor='w')
        
        notification_frame = tk.Frame(settings_frame, bg='white')
        notification_frame.pack(fill='x', padx=15, pady=10)
        
        notification_var = tk.BooleanVar(value=True)
        notification_check = tk.Checkbutton(notification_frame, 
                                           text="Hiển thị thông báo cảnh báo",
                                           variable=notification_var, bg='white', 
                                           font=('Segoe UI', 9))
        notification_check.pack(anchor='w')
        
        button_frame = tk.Frame(settings_frame, bg='white')
        button_frame.pack(fill='x', padx=15, pady=15)
        
        def save_settings():
            messagebox.showinfo("Thông báo", "Cài đặt đã được lưu!")
            settings_window.destroy()
        
        save_btn = tk.Button(button_frame, text="Lưu cài đặt", command=save_settings,
                           bg='#3b82f6', fg='white', font=('Segoe UI', 9, 'bold'),
                           padx=20, pady=8, relief='flat')
        save_btn.pack(side='right')
        
        cancel_btn = tk.Button(button_frame, text="Hủy", command=settings_window.destroy,
                             bg='#6b7280', fg='white', font=('Segoe UI', 9),
                             padx=20, pady=8, relief='flat')
        cancel_btn.pack(side='right', padx=(0, 10))

def main():
    root = tk.Tk()
    
    def toggle_monitoring_key(event):
        app.toggle_monitoring()
    
    def exit_app_key(event):
        if app.is_monitoring:
            app.stop_monitoring()
        root.quit()
    
    app = PostureMonitoringGUI(root)
    
    root.bind('<F1>', toggle_monitoring_key)  # F1 để toggle monitoring
    root.bind('<Control-q>', exit_app_key)    # Ctrl+Q để thoát
    root.bind('<Escape>', lambda e: app.stop_monitoring() if app.is_monitoring else None)
    
    def on_closing():
        if app.is_monitoring:
            app.stop_monitoring()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.focus_set()
    
    root.mainloop()

if __name__ == "__main__":
    main()