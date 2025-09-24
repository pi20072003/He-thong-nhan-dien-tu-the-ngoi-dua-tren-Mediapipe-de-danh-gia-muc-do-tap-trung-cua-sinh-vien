#Giao diện giám sát tư thế thông minh

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

# Import các thư viện cần thiết
try:
    import mediapipe as mp
    from tensorflow.keras.models import load_model
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False # type: ignore

class PostureMonitoringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Giám Sát Tư Thế")

        # lưu (thời gian, tư thế, % tập trung)
        self.history = []  
        self.focus_scores = {
                "ngoi thang": 100,  "guc dau": 40,  "nga nguoi": 60,
                "quay trai": 70,    "quay phai": 70,    "chong tay": 80,
                "unknown": 0 }
        # Tự động điều chỉnh kích thước theo màn hình
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
        
        self.last_alert_time = None
        self.alert_cooldown_seconds = 1  # Chỉ cảnh báo 1 lần mỗi giây
        
        self.posture_buffer = deque(maxlen=10) # Buffer để làm mượt dự đoán
        # Lưu trữ dữ liệu
        self.posture_history = deque(maxlen=100)
        self.recent_alerts = deque(maxlen=10)
        self.daily_stats = {"good_time": 0, "bad_time": 0, "alerts": 0}
        
        # Các thành phần mô hình
        self.model = None
        self.scaler = None
        self.labels = {"0": "ngoi thang", "1": "guc dau", "2": "nga nguoi",
                       "3": "quay trai", "4": "quay phai", "5": "chong tay"}
        
        # Camera
        self.cap = None
        self.current_frame = None
        
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
        
        # Danh sách lưu trữ các nhãn cảnh báo
        self.alert_labels = []
    
    def setup_window_size(self):
        """Tự động điều chỉnh kích thước cửa sổ theo màn hình"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Tính toán kích thước cửa sổ (85% màn hình, tối đa 1300x850)
        max_width = min(int(screen_width * 0.85), 1300)
        max_height = min(int(screen_height * 0.8), 850)
        
        # Đảm bảo kích thước tối thiểu
        width = max(max_width, 1100)
        height = max(max_height, 700)
        
        # Căn giữa cửa sổ
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(1100, 700)
        self.root.resizable(True, True)
        
        # Lưu kích thước để sử dụng trong layout
        self.window_width = width
        self.window_height = height
    
    def setup_gui(self):
        # Main container
        main_container = tk.Frame(self.root, bg='#f8fafc')
        main_container.pack(fill='both', expand=True)

        self.create_header(main_container)

        # Không tạo canvas và scrollbar nữa, chỉ dùng frame thường
        content_frame = tk.Frame(main_container, bg='#f8fafc')
        content_frame.pack(fill='both', expand=True)

        # Content
        self.create_main_content(content_frame)
    
    def create_header(self, parent):
        """Tạo header compact với trạng thái làm việc"""
        header_frame = tk.Frame(parent, bg='#f8fafc')
        header_frame.pack(fill='x', padx=20, pady=0)
        
        # Title
        title_label = tk.Label(header_frame, text="Hệ thống giám sát ", 
                              font=('Segoe UI', 18, 'bold'), 
                              fg='#1f2937', bg='#f8fafc')
        title_label.pack(side='left')
        
        # Subtitle
        subtitle_label = tk.Label(header_frame, text="Giám sát tư thế thông minh", 
                                 font=('Segoe UI', 10), 
                                 fg='#6b7280', bg='#f8fafc')
        subtitle_label.pack(side='left', padx=(15, 0))
    
    def create_main_content(self, parent):
        """Tạo nội dung chính với camera CỐ ĐỊNH và bố cục mới"""
        content_frame = tk.Frame(parent, bg='#f8fafc')
        content_frame.pack(fill='both', expand=True, padx=20, pady=0)
        
        # Left column (camera)
        left_column = tk.Frame(content_frame, bg='#f8fafc')
        left_column.pack(side='left', fill='y', padx=(0, 15))
        self.setup_camera_section(left_column)
        
        # Right column (cảnh báo + thống kê phiên làm việc)
        right_column = tk.Frame(content_frame, bg='#f8fafc')
        right_column.pack(side='left', fill='y', padx=(0, 15))
        
        # KHÔNG tạo status_frame ở đây nữa
        
        # Ô cảnh báo
        self.setup_alerts_panel(right_column)
        
        # Ô thống kê phiên làm việc
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
        
        # Ô cài đặt
        self.setup_settings_panel(right_column)
    
    def setup_camera_section(self, parent):
        """Camera section với vùng CỐ ĐỊNH TUYỆT ĐỐI"""
         # Camera frame
        camera_frame = self.create_card(parent, "Giám sát trực tiếp")
        camera_frame.pack(fill='x', pady=(0, 15))
        
        # Control buttons
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

        # Hiển thị trạng thái 
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

        # Camera container - CỐ ĐỊNH TUYỆT ĐỐI
        camera_container_outer = tk.Frame(camera_frame, bg='white')
        camera_container_outer.pack(fill='x', padx=15, pady=(0, 15))
        
        # Tính kích thước camera tối ưu nhưng CỐ ĐỊNH
        base_width = min(int(self.window_width * 0.5), 700)  # Tối đa 700px
        base_height = int(base_width * 3 / 4)  # 4:3 ratio
        
        # Container camera với kích thước CỐ ĐỊNH TUYỆT ĐỐI
        self.camera_container = tk.Frame(camera_container_outer, 
                                        bg='#1f2937', 
                                        relief='solid', bd=2,
                                        width=base_width, 
                                        height=base_height)
        self.camera_container.pack(anchor='center')  # Căn giữa
        self.camera_container.pack_propagate(False)  # QUAN TRỌNG: Không thay đổi kích thước
        
        # Label camera - đặt CỐ ĐỊNH bằng place()
        self.camera_label = tk.Label(self.camera_container, 
                                    text="Camera dừng\nBấm 'Bắt đầu giám sát' để khởi động",
                                    bg='#1f2937', fg='white',
                                    font=('Segoe UI', 12),
                                    anchor='center')
        self.camera_label.place(x=0, y=0, width=base_width, height=base_height)
        
        # Lưu kích thước để dùng trong processing
        self.camera_width = base_width
        self.camera_height = base_height
        
        # Status section
        status_frame = tk.Frame(camera_frame, bg='white')
        status_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        # Posture status
        posture_frame = tk.Frame(status_frame, bg='#f3f4f6', relief='flat', bd=1)
        posture_frame.pack(side='left', fill='both', expand=True, padx=(0, 8))
        
        tk.Label(posture_frame, text="Tư thế hiện tại", 
                font=('Segoe UI', 9), fg='#6b7280', bg='#f3f4f6').pack(pady=(12, 3))
        
        self.posture_label = tk.Label(posture_frame, text="Chưa xác định", 
                                     font=('Segoe UI', 11, 'bold'),
                                     fg='#374151', bg='#f3f4f6')
        self.posture_label.pack(pady=(0, 12))
        
        # Confidence status
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
        
        # Time stat
        time_frame = self.create_stat_box(stats_container, "00:00:00", "Thời gian làm việc", 
                                         '#3b82f6', '#dbeafe')
        time_frame.pack(side='left', fill='both', expand=True, padx=(0, 8))
        self.session_time_label = time_frame.children['!label']
        
        # Good posture stat
        good_frame = self.create_stat_box(stats_container, "0%", "Tư thế tốt", 
                                         '#10b981', '#d1fae5')
        good_frame.pack(side='left', fill='both', expand=True, padx=(4, 4))
        self.good_posture_label = good_frame.children['!label']
        
        # Alerts stat
        alert_frame = self.create_stat_box(stats_container, "0", "Cảnh báo", 
                                          '#ef4444', '#fee2e2')
        alert_frame.pack(side='left', fill='both', expand=True, padx=(8, 0))
        self.alerts_count_label = alert_frame.children['!label']
    
    def setup_right_column_scroll(self, parent):
        """Cột phải với scroll"""
        # Canvas cho scroll
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
        
        # Mouse wheel scroll
        def _on_right_mousewheel(event):
            right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        right_canvas.bind("<MouseWheel>", _on_right_mousewheel)
        
        # Right column content
        self.setup_quick_stats(right_scrollable)
        self.setup_recent_history(right_scrollable)
        self.setup_settings_panel(right_scrollable)
    
    def setup_alerts_panel(self, parent):
        """Panel cảnh báo"""
        alerts_frame = self.create_card(parent, "Cảnh báo")
        alerts_frame.pack(fill='x', pady=(0, 12))
        
        self.alerts_container = tk.Frame(alerts_frame, bg='white', width=220, height=200)
        self.alerts_container.pack(fill='x', padx=15, pady=(0, 15))
        self.alerts_container.pack_propagate(False)
        
        # Default message
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
        
        # Độ nhạy cảnh báo
        self.sensitivity_var = tk.StringVar(value="Trung bình (≥70%)")
        tk.Label(settings_container, text="Độ nhạy cảnh báo:", 
                font=('Segoe UI', 10), fg='#374151', bg='white').pack(anchor='w', pady=(2, 2))
        self.sensitivity_combo = ttk.Combobox( settings_container, textvariable=self.sensitivity_var,
                                            values=["Thấp (≥50%)", "Trung bình (≥70%)", "Cao (≥90%)"],
                                            state="readonly")
        self.sensitivity_combo.pack(fill='x', pady=(0, 10))
        
        # Buttons
        buttons = [
            ("Xuất báo cáo", self.export_report, '#059669'),
            ("Cài đặt Camera", self.camera_settings, '#7c3aed')]
        
        for text, command, color in buttons:
            btn = tk.Button(settings_container, text=text, command=command,
                           bg=color, fg='white', font=('Segoe UI', 8),
                           padx=12, pady=4, relief='flat')
            btn.pack(fill='x', pady=1)
    
    def create_card(self, parent, title):
        """Tạo card với header"""
        card = tk.Frame(parent, bg='white', relief='solid', bd=1)
        
        # Header
        header = tk.Frame(card, bg='white')
        header.pack(fill='x', padx=15, pady=(12, 8))
        
        tk.Label(header, text=title, font=('Segoe UI', 10, 'bold'),
                fg='#1f2937', bg='white').pack(side='left')
        return card
    
    def create_stat_box(self, parent, value, label, color, bg_color):
        """Tạo stat box"""
        box = tk.Frame(parent, bg=bg_color, relief='flat', bd=1)
        
        # Value
        value_label = tk.Label(box, text=value, font=('Segoe UI', 14, 'bold'),
                              fg=color, bg=bg_color)
        value_label.pack(pady=(10, 3))
        
        # Label
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

            self.is_monitoring = True
            self.session_start_time = time.time()

            self.history = []  # reset lịch sử tư thế
            self.posture_buffer.clear()
            self.session_time = 0
            self.good_posture_time = 0
            self.alerts_count = 0
            self.last_alert_time = None # Reset thời gian cảnh báo
            self.session_start_datetime = datetime.now()  # lưu thời gian bắt đầu

            if not self.sensitivity_var.get():
                self.sensitivity_var.set("Trung bình (≥70%)")
            self.sensitivity_combo.config(state="disabled") # Khóa combobox độ nhạy khi bắt đầu giám sát

            if self.current_posture is not None:
                self.history.append((0, self.current_posture))  
            else:
                self.history.append((0, "unknown")) # Nếu chưa có tư thế nào gán unknown
                
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
        self.is_monitoring = False
        
        if self.cap:
            self.cap.release()
        
        self.sensitivity_combo.config(state="readonly") #Mở lại combobox để chọn lại độ nhạy

        # Update UI
        self.start_btn.config(text="Bắt đầu giám sát", bg='#3b82f6')
        self.status_indicator.config(fg='#ef4444')
        self.status_text.config(text="Dừng")
        self.camera_label.config(image="", text="Camera dừng\nBấm 'Bắt đầu giám sát' để khởi động")
        self.posture_label.config(text="Chưa xác định")
        self.confidence_label.config(text="0.0%")
        self.session_end_datetime = datetime.now()  # lưu thời gian kết thúc
    
    def camera_loop(self):
        """Vòng lặp xử lý camera"""
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
                
                # Draw pose landmarks
                if results.pose_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
                    # Predict posture
                    features = self.extract_features(results.pose_landmarks)
                    if features is not None:
                        posture, confidence = self.predict_posture(features)
                        self.current_posture = posture
                        self.confidence = confidence # Vẫn hiển thị confidence của frame hiện tại
                        
                        # --- Cải tiến logic làm mượt dự đoán ---
                        self.posture_buffer.append(posture)
                        
                        # Chỉ đưa ra quyết định khi buffer đã đầy để có đủ dữ liệu
                        if len(self.posture_buffer) == self.posture_buffer.maxlen:
                            try:
                                # Tìm tư thế ổn định (xuất hiện nhiều nhất)
                                stable_posture = max(set(self.posture_buffer), key=self.posture_buffer.count)
                                count = self.posture_buffer.count(stable_posture)
                                
                                # Ngưỡng ổn định: tư thế phải chiếm > 60% buffer
                                STABILITY_THRESHOLD = 6 

                                # Chỉ cập nhật nếu tư thế ổn định đã thay đổi và đạt ngưỡng
                                if count > STABILITY_THRESHOLD and stable_posture != self.current_posture:
                                    sensitivity_text = self.sensitivity_var.get() if hasattr(self, "sensitivity_var") else "Trung bình"
                                    if "50" in sensitivity_text:
                                        threshold = 0.50
                                    elif "90" in sensitivity_text:
                                        threshold = 0.90
                                    else:
                                        threshold = 0.70
                                    if self.confidence >= threshold:  # ✅ chỉ ghi nếu độ tin cậy >= ngưỡng
                                        session_time = int(time.time() - self.session_start_time)
                                        self.history.append((session_time, stable_posture))
                                        self.current_posture = stable_posture

                            except ValueError:
                                pass # Bỏ qua nếu buffer rỗng

                        # Update UI
                        self.root.after(0, self.update_posture_display)
                        
                        # Check for alerts
                        self.check_alerts(self.current_posture) # Cảnh báo dựa trên tư thế ổn định
                
                # Convert và hiển thị với kích thước CỐ ĐỊNH
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Sử dụng kích thước cố định của camera container
                display_width = self.camera_width - 10  # Trừ padding
                display_height = self.camera_height - 10
                
                # Resize và crop để fit
                original_width, original_height = frame_pil.size
                scale_w = display_width / original_width
                scale_h = display_height / original_height
                scale = max(scale_w, scale_h)
                
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                frame_pil = frame_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Crop from center
                left = (new_width - display_width) // 2
                top = (new_height - display_height) // 2
                right = left + display_width
                bottom = top + display_height
                frame_pil = frame_pil.crop((left, top, right, bottom))
                frame_tk = ImageTk.PhotoImage(frame_pil)
                self.root.after(0, lambda img=frame_tk: self.update_camera_display(img))
                time.sleep(0.03)  # ~30 FPS
    
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
        self.camera_label.image = image  # Keep reference
    
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
        
        # Cập nhật độ tin cậy
        confidence_pct = self.confidence * 100
        self.confidence_label.config(text=f"{confidence_pct:.1f}%")
        
        # Cập nhật màu nhãn tư thế
        if self.current_posture == "ngoi thang":
            self.posture_label.config(fg='#10b981')  # Xanh lá
        else:
            self.posture_label.config(fg='#ef4444')  # Đỏ
    
    def check_alerts(self, posture):
        """Kiểm tra và tạo cảnh báo"""
        current_time = datetime.now()
        
        sensitivity_text = self.sensitivity_var.get()
        if "50" in sensitivity_text:
            threshold = 0.50
        elif "90" in sensitivity_text:
            threshold = 0.90
        else:
            threshold = 0.70  # mặc định Trung bình

        if self.confidence < threshold: #độ tin cậy không đạt ngưỡng thì bỏ qua cảnh báo
            return
    
        # Chỉ cảnh báo nếu đã qua thời gian cooldown
        if self.last_alert_time and \
           (current_time - self.last_alert_time).total_seconds() < self.alert_cooldown_seconds:
            return  # Bỏ qua nếu chưa đủ thời gian

        self.last_alert_time = current_time  # Cập nhật thời gian cảnh báo cuối

        # Tên cảnh báo tiếng Việt
        alert_names = {
            "guc dau": "Tư thế không tốt: Cúi đầu",
            "nga nguoi": "Tư thế không tốt: Ngả người",
            "ngoi thang": "Tư thế tốt: Ngồi thẳng",
            "quay trai": "Tư thế không tốt: Quay trái",
            "quay phai": "Tư thế không tốt: Quay phải",
            "chong tay": "Tư thế không tốt: Chống tay"
        }

        alert_message = alert_names.get(posture, f"Tư thế: {posture}")

        # Thêm vào danh sách cảnh báo gần đây
        self.recent_alerts.appendleft((current_time, alert_message))
        if posture != "ngoi thang" and self.confidence >= threshold:
            self.alerts_count += 1  # Chỉ tăng số cảnh báo với tư thế xấu

        # Cập nhật hiển thị cảnh báo
        self.root.after(0, self.update_alerts_display)
    
    def update_alerts_display(self):
        """Cập nhật panel cảnh báo mà không destroy toàn bộ widget"""
        max_alerts = 3
        alerts = list(self.recent_alerts)[:max_alerts]

        # Nếu chưa có đủ label, tạo thêm
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

        # Cập nhật nội dung các label
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

        # Nếu không có cảnh báo, hiển thị label mặc định
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
            if self.is_monitoring and self.session_start_time:
                self.session_time = int(time.time() - self.session_start_time)
                
                # Cập nhật thời gian tư thế tốt
                if self.current_posture == "ngoi thang":
                    self.good_posture_time += 1
                
                # Cập nhật hiển thị
                self.update_session_stats()
            
            # Lên lịch cập nhật tiếp theo
            self.root.after(1000, update_timer)
        update_timer()
    
    def update_session_stats(self):
        """Cập nhật thống kê phiên làm việc"""
        # Định dạng thời gian phiên
        hours = self.session_time // 3600
        minutes = (self.session_time % 3600) // 60
        seconds = self.session_time % 60
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        self.session_time_label.config(text=time_str)
        
        # Tính phần trăm tư thế tốt
        if self.session_time > 0:
            good_percentage = (self.good_posture_time / self.session_time) * 100
            self.good_posture_label.config(text=f"{good_percentage:.0f}%")
        
        # Cập nhật số lượng cảnh báo
        self.alerts_count_label.config(text=str(self.alerts_count))
    
    def toggle_fullscreen(self):
        """Chuyển đổi chế độ toàn màn hình"""
        try:
            if self.root.attributes('-fullscreen'):
                self.root.attributes('-fullscreen', False)
                self.root.state('zoomed')
            else:
                self.root.attributes('-fullscreen', True)
        except:
            # Fallback cho hệ thống không hỗ trợ
            try:
                self.root.state('zoomed')
            except:
                pass
    
    def export_report(self):
        """Xuất báo cáo"""
        from tkinter import filedialog
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write("Báo cáo tư thế\n")
                    f.write(f"(Xuất lúc: {now_str})\n")
                    if hasattr(self, "session_start_datetime"):
                        f.write(f"Thời gian bắt đầu giám sát: {self.session_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    if hasattr(self, "session_end_datetime"):
                        f.write(f"Thời gian kết thúc giám sát: {self.session_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Độ nhạy sử dụng: {self.sensitivity_var.get()}\n")
                    f.write(f"Thời gian giám sát: {self.session_time // 3600:02d}:{(self.session_time % 3600) // 60:02d}:{self.session_time % 60:02d}\n")
                    f.write(f"Số cảnh báo: {self.alerts_count}\n")
                    last_posture = getattr(self, "current_posture", "Chưa giám sát")
                    f.write(f"Tư thế cuối cùng phát hiện: {last_posture}\n")

                    # Tính trung bình mức độ tập trung dựa trên toàn bộ lịch sử
                    focus_score_avg = 0
                    if self.history:
                        total_time = self.session_time
                        weighted_sum = 0
                        last_time = 0
                        
                        # Lặp qua lịch sử để tính điểm cho từng khoảng thời gian
                        for i in range(len(self.history) - 1):
                            current_ts, current_posture = self.history[i]
                            next_ts, _ = self.history[i+1]
                            duration = next_ts - current_ts
                            score = self.focus_scores.get(current_posture, 0)
                            weighted_sum += score * duration
                        
                        # Tính cho tư thế cuối cùng đến hết phiên
                        last_ts, last_posture = self.history[-1]
                        duration = total_time - last_ts
                        score = self.focus_scores.get(last_posture, 0)
                        weighted_sum += score * duration

                        focus_score_avg = weighted_sum / total_time if total_time > 0 else 0
                    else:
                        focus_score_avg = self.focus_scores.get(self.current_posture, 0)
                    f.write(f"Mức độ tập trung trung bình: {focus_score_avg:.2f}%\n")
                    f.write("\nLỊCH SỬ THAY ĐỔI TƯ THẾ:\n")
                    f.write("-" * 40 + "\n")
                    if not getattr(self, "history", []):
                        f.write("Không có dữ liệu lịch sử để hiển thị.\n")
                    else:
                        for ts, posture in self.history:
                            h, remainder = divmod(ts, 3600)
                            m, s = divmod(remainder, 60)
                            f.write(f"Thời gian {h:02d}:{m:02d}:{s:02d} - Tư thế: {posture}\n")

                messagebox.showinfo("Xuất báo cáo", f"Báo cáo đã được lưu: {file_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xuất báo cáo: {e}")
    
    def camera_settings(self):
        """Cài đặt camera"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Cài đặt Camera")
        settings_window.geometry("350x200")
        settings_window.resizable(False, False)
        settings_window.configure(bg='#f8fafc')
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Camera ID setting
        tk.Label(settings_window, text="ID Camera:", font=('Segoe UI', 10, 'bold'),
                fg='#374151', bg='#f8fafc').pack(pady=(20, 5))
        camera_id_var = tk.StringVar(value="0")
        camera_spinbox = tk.Spinbox(settings_window, from_=0, to=9, width=10, 
                                   textvariable=camera_id_var)
        camera_spinbox.pack(pady=(0, 20))
        
        # Test button
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
   - Cúi đầu: Cần điều chỉnh
   - Ngả người: Cần điều chỉnh
   - Quay trái/phải: Cần điều chỉnh
   - Chống tay: Cần điều chỉnh
4. Cách tính điểm tập trung:
   - Ngồi thẳng: 100 %      - Cúi đầu: 40 %
   - Ngả người: 60 %        - Quay trái/phải: 70 %   
   - Chống tay: 80 % 
   - Thời gian tư thế cần tính 
        = tgian bắt đầu tư thế sau - tgian bắt đầu tư thế cần tính
Lưu ý: Ngồi cách camera 60-100cm để đạt độ chính xác tốt nhất."""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Hướng dẫn sử dụng")
        help_window.geometry("500x400")
        help_window.configure(bg='#f8fafc')
        help_window.transient(self.root)
        help_window.grab_set()
        
        # Text widget với scrollbar
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
        about_text = """Hệ thống giám sát
Hệ thống giám sát tư thế thông minh sử dụng AI
để phân tích và cảnh báo tư thế ngồi sai.
Công nghệ sử dụng:
• MediaPipe - Nhận diện khung xương người
• TensorFlow - Mô hình AI phân loại tư thế  
• OpenCV - Xử lý hình ảnh camera
• Python & Tkinter - Giao diện người dùng
Phát triển bởi: Đỗ Quang Huy - pi2007
                          Lê Quang Huy - playmaker
"""
        messagebox.showinfo("Thông tin Hệ thống giám sát ", about_text)
    
    def open_settings(self):
        """Mở cài đặt tổng quát"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Cài đặt")
        settings_window.geometry("400x300")
        settings_window.resizable(False, False)
        settings_window.configure(bg='#f8fafc')
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Title
        title_label = tk.Label(settings_window, text="Cài đặt Hệ thống giám sát ", 
                              font=('Segoe UI', 14, 'bold'), 
                              fg='#1f2937', bg='#f8fafc')
        title_label.pack(pady=(20, 15))
        
        # Settings frame
        settings_frame = tk.Frame(settings_window, bg='white', relief='solid', bd=1)
        settings_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Alert sensitivity
        sensitivity_frame = tk.Frame(settings_frame, bg='white')
        sensitivity_frame.pack(fill='x', padx=15, pady=15)
        
        tk.Label(sensitivity_frame, text="Độ nhạy cảnh báo:", 
                font=('Segoe UI', 10), fg='#374151', bg='white').pack(anchor='w')
        
        sensitivity_var = tk.StringVar(value="Trung bình")
        sensitivity_combo = ttk.Combobox(sensitivity_frame, textvariable=sensitivity_var,
                                       values=["Thấp", "Trung bình", "Cao"], state="readonly")
        sensitivity_combo.pack(fill='x', pady=(5, 0))
        
        # Auto start
        auto_start_frame = tk.Frame(settings_frame, bg='white')
        auto_start_frame.pack(fill='x', padx=15, pady=10)
        
        auto_start_var = tk.BooleanVar()
        auto_start_check = tk.Checkbutton(auto_start_frame, 
                                         text="Tự động bắt đầu giám sát khi khởi động",
                                         variable=auto_start_var, bg='white', 
                                         font=('Segoe UI', 9))
        auto_start_check.pack(anchor='w')
        
        # Notification
        notification_frame = tk.Frame(settings_frame, bg='white')
        notification_frame.pack(fill='x', padx=15, pady=10)
        
        notification_var = tk.BooleanVar(value=True)
        notification_check = tk.Checkbutton(notification_frame, 
                                           text="Hiển thị thông báo cảnh báo",
                                           variable=notification_var, bg='white', 
                                           font=('Segoe UI', 9))
        notification_check.pack(anchor='w')
        
        # Buttons
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
    
    # Keyboard shortcuts
    def toggle_monitoring_key(event):
        app.toggle_monitoring()
    
    def exit_app_key(event):
        if app.is_monitoring:
            app.stop_monitoring()
        root.quit()
    
    app = PostureMonitoringGUI(root)
    
    # Bind shortcuts
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