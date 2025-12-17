#Giao diện giám sát tư thế thông minh
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import json
import os
from datetime import datetime   
import pickle
from collections import deque
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import serial

try:
    import mediapipe as mp
    from tensorflow.keras.models import load_model
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False # type: ignore

# Kết nối serial
try:
    ser = serial.Serial('COM3', 115200, timeout=1)
    time.sleep(2)  # Đợi STM32 khởi động
    SERIAL_AVAILABLE = True
except:
    print("Không thể kết nối serial")
    SERIAL_AVAILABLE = False

class PostureMonitoringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Giám Sát Tư Thế")

        # lưu (thời gian, tư thế, % tập trung)
        self.history = []  # Lịch sử tư thế trong phiên làm việc
        self.log_records = []  # Lưu log chi tiết từng giây

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
        self.pie_canvas = None
        self.longest_good_streak = 0
        self.current_good_streak = 0

        self.current_led_state = 'O'  # Trạng thái LED hiện tại
        self.last_posture_change_time = 0  # Thời điểm tư thế thay đổi

        # Tính thời gian duy trì tư thế
        self.good_stable_time = 0
        self.bad_time = 0
        self.posture_start_time = None
        self.last_stable_posture = None


        self.last_alert_time = None
        self.alert_cooldown_seconds = 1  # Chỉ cảnh báo 1 lần mỗi giây
        
        self.posture_buffer = deque(maxlen=10) # Buffer để làm mượt dự đoán
        # Lưu trữ dữ liệu
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
        
        # Danh sách lưu trữ các nhãn cảnh báo
        self.alert_labels = []
    
    # Hàm gửi UART
    def send_led_command(self, cmd):
        """Gửi lệnh LED qua UART - chỉ gửi khi thay đổi"""
        if not SERIAL_AVAILABLE:
            return
        
        # Chỉ gửi khi trạng thái thay đổi
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
        
        # kích thước cửa sổ (85% màn hình, tối đa 1300x850)
        max_width = min(int(screen_width * 0.85), 1300)
        max_height = min(int(screen_height * 0.8), 850)
        
        # Đảm bảo kích thước tối thiểu
        width = max(max_width, 1200)
        height = max(max_height, 750)
        
        # Căn giữa cửa sổ
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(1200, 750)
        self.root.maxsize(1200, 750)
        self.root.resizable(False, False)
        
        # Lưu kích thước để sử dụng trong layout
        self.window_width = width
        self.window_height = height
    
    def setup_gui(self):
        # Main container
        main_container = tk.Frame(self.root, bg='#f8fafc')
        main_container.pack(fill='both', expand=True)

        self.create_header(main_container)

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
        
        # Right column (cảnh báo + thống kê phiên làm việc + cài đặt)
        right_column = tk.Frame(content_frame, bg='#f8fafc')
        right_column.pack(side='left', fill='y', expand=True)

        # Chia thành 2 cột nhỏ: thống kê + cài đặt | cảnh báo
        left_right_split = tk.Frame(right_column, bg='#f8fafc')
        left_right_split.pack(fill='both', expand=True)

        # Cột trái trong phần phải: Thống kê + Cài đặt
        stats_settings_col = tk.Frame(left_right_split, bg='#f8fafc')
        stats_settings_col.pack(side='left', fill='y', padx=(0, 15))

        # Ô thống kê phiên làm việc
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

        # Ô cài đặt
        self.setup_settings_panel(stats_settings_col)

        # Ô cảnh báo
        alerts_col = tk.Frame(left_right_split, bg='#f8fafc')
        alerts_col.pack(side='left', fill='y')
        self.setup_alerts_panel(alerts_col)
        
# Khung biểu đồ phân tích (dưới ô cảnh báo)
        # Frame ngoài có viền
        chart_card = tk.Frame(alerts_col, bg="white", relief="solid", bd=1)
        chart_card.pack(fill="x", pady=(0, 15))

        # Tiêu đề nằm bên trong viền
        chart_header = tk.Frame(chart_card, bg="white")
        chart_header.pack(fill="x", padx=15, pady=(10, 5))

        tk.Label(
            chart_header, text="Biểu đồ đánh giá",
            font=("Segoe UI", 10, "bold"),
            bg="white", fg="#000000",
            anchor="w").pack(side="left")

        # Khung chứa biểu đồ
        self.chart_border = tk.Frame(chart_card, bg="white", height=185, width=330)
        self.chart_border.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        self.chart_border.pack_propagate(False)

        # Khung con chứa nội dung biểu đồ
        self.analysis_frame = tk.Frame(self.chart_border, bg="white")
        self.analysis_frame.pack(fill="both", expand=True, padx=2, pady=2)
        self.analysis_frame.pack_propagate(False)

        # Placeholder khi chưa có file log
        self.analysis_placeholder = tk.Label(
            self.analysis_frame,
            text="Thêm file log để đánh giá",
            font=('Segoe UI', 9, 'italic'),
            bg='white', fg='gray'
        )
        self.analysis_placeholder.pack(pady=20)
        

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

        # Camera container
        camera_container_outer = tk.Frame(camera_frame, bg='white')
        camera_container_outer.pack(fill='x', padx=15, pady=(0, 15))
        
        # kích thước camera 
        base_width = min(int(self.window_width * 0.5), 700)  # Tối đa 700px
        base_height = int(base_width * 3 / 4)  # 4:3 ratio
        
        # Container camera với kích thước cố định
        self.camera_container = tk.Frame(camera_container_outer, 
                                        bg='#1f2937', 
                                        relief='solid', bd=2,
                                        width=base_width, 
                                        height=base_height)
        self.camera_container.pack(anchor='center')  # Căn giữa
        self.camera_container.pack_propagate(False)  # Không thay đổi kích thước
        
        # Label camera - đặt cố định bằng place()
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

    def setup_alerts_panel(self, parent):
        """Panel cảnh báo"""
        alerts_frame = self.create_card(parent, "Cảnh báo")
        alerts_frame.pack(fill='x', pady=(0, 15))
        
        self.alerts_container = tk.Frame(alerts_frame, bg='white', width=220, height=249)
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
        # Tạo nút xuất báo cáo có thể khóa/mở
        self.export_btn = tk.Button(settings_container, text="Xuất báo cáo", 
                                    command=self.export_report,
                                    bg='#059669', fg='white', font=('Segoe UI', 8, 'bold'),
                                    padx=12, pady=4, relief='flat')
        self.export_btn.pack(fill='x', pady=1)

        # Nút Xuất file log có thể khóa/mở
        self.export_log_btn =  tk.Button(settings_container, text="Xuất file log",
                                            command=self.export_log,
                                            bg="#3b82f6", fg="white", font=('Segoe UI', 8, 'bold'),
                                            padx=12, pady=4, relief='flat')
        self.export_log_btn.pack(fill='x', pady=1)

        # Nút duyệt file log (Browse) có thể khóa/mở
        self.browse_log_btn = tk.Button(settings_container, text="Browse file log",
                                        command=self.browse_log_file,
                                        bg='#e5e7eb', fg='black', font=('Segoe UI', 8, 'bold'),
                                        padx=12, pady=4, relief='flat')
        self.browse_log_btn.pack(fill='x', pady=1)

        # Nút cài đặt camera
        self.camera_btn = tk.Button(settings_container, text="Cài đặt Camera", 
                                    command=self.camera_settings,
                                    bg='#7c3aed', fg='white', font=('Segoe UI', 8, 'bold'),
                                    padx=12, pady=4, relief='flat')
        self.camera_btn.pack(fill='x', pady=1)
    
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
            # Mở camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở camera!")
                return

            # Thiết lập thời gian bắt đầu
            self.session_start_datetime = datetime.now()
            self.session_start_time = time.time()

            # Reset trạng thái phiên
            self.is_monitoring = True
            self.history = []
            self.posture_buffer.clear()
            self.log_records.clear()
            self.session_time = 0
            self.good_posture_time = 0
            self.alerts_count = 0
            self.last_alert_time = None
            self.bad_posture_start_time = None
            self.good_posture_start_time = None
            self.last_stable_posture = None

            # RESET LED khi bắt đầu
            self.current_led_state = 'O'
            self.send_led_command('O')
            # Reset bộ đếm thời gian
            self.good_stable_time = 0
            self.bad_time = 0

            # Reset alert panel khi bắt đầu:
            self.recent_alerts.clear()
            self.update_alerts_display()

            # Reset chart
            for widget in self.analysis_frame.winfo_children():
                widget.destroy()
            # Xóa canvas cũ nếu tồn tại
            if self.pie_canvas:
                self.pie_canvas = None

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
        # Ghi thời điểm dừng 1 lần ở đây
        self.session_end_datetime = datetime.now()
        self.monitor_stop_time = time.time()

        self.is_monitoring = False
        if self.cap:
            self.cap.release()

        # Mở lại các control
        self.sensitivity_combo.config(state="readonly")
        self.export_btn.config(state="normal")
        self.export_log_btn.config(state="normal")
        self.browse_log_btn.config(state="normal")

        # TẮT LED khi dừng
        self.send_led_command('O')
        self.current_led_state = 'O'
        self.good_stable_time = 0
        self.bad_time = 0
        self.bad_posture_start_time = None
        self.good_posture_start_time = None
        self.last_stable_posture = None
        self.current_led_state = 'O'
        self.send_led_command('O')

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

                # ===== KIỂM TRA NGAY: KHÔNG THẤY NGƯỜI =====
                if not results.pose_landmarks:
                    self.send_led_command('W') # GỬI W khi không thấy người
                    self.good_posture_start_time = None
                    self.bad_posture_start_time = None
                    
                    # Cập nhật trạng thái
                    self.current_posture = "khong thay nguoi"
                    self.confidence = 0.0
                    self.posture_buffer.append("khong thay nguoi")

                    # Ghi vào lịch sử nếu trạng thái thay đổi
                    if len(self.posture_buffer) >= 5 and self.posture_buffer.count("khong thay nguoi") >= 3:
                        session_time = self.session_time
                        if not self.history or self.history[-1][1] != "khong thay nguoi":
                            self.history.append((session_time, "khong thay nguoi"))
                            print(f"Đã thêm vào history: {session_time}s - khong thay nguoi")
                            if not self.log_records or self.log_records[-1][0] != session_time:
                                self.log_records.append((session_time, "khong thay nguoi"))
                    
                    # Gọi kiểm tra cảnh báo
                    self.check_alerts("khong thay nguoi")
                    
                    # Cập nhật hiển thị
                    self.root.after(0, self.update_posture_display)
                    
                    # Xử lý hiển thị camera (giữ nguyên)
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
                    continue 

                # ===== XỬ LÝ KHI CÓ PHÁT HIỆN NGƯỜI =====
                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                features = self.extract_features(results.pose_landmarks)
                if features is not None:
                    posture, confidence = self.predict_posture(features)
                    self.current_posture = posture
                    self.confidence = confidence

                    self.posture_buffer.append(posture)

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
                                session_time = self.session_time

                                # thêm vào history khi tư thế ổn định thay đổi
                                should_add_to_history = False
                                if not self.history:
                                    should_add_to_history = True
                                else:
                                    if (isinstance(self.history[-1], (list, tuple)) and len(self.history[-1]) == 2):
                                        last_timestamp, last_posture = self.history[-1]
                                        time_diff = session_time - last_timestamp
                                        if stable_posture != last_posture and time_diff >= 2:
                                            should_add_to_history = True
                                    else:
                                        should_add_to_history = True

                                if should_add_to_history:
                                    self.history.append((session_time, stable_posture))
                                    if not self.log_records or self.log_records[-1][0] != session_time:
                                        self.log_records.append((session_time, stable_posture))
                                    print(f"Đã thêm vào history: {session_time}s - {stable_posture}")

                                self.current_posture = stable_posture
                                
                                # ===== ĐIỀU KHIỂN LED =====
                                current_time = time.time()

                                # Phân loại tư thế
                                is_good_posture = (stable_posture == "ngoi thang")
                                is_no_person = (stable_posture == "khong thay nguoi")
                                is_bad_posture = not is_good_posture and not is_no_person

                                # ======== TƯ THẾ TỐT ========
                                if is_good_posture:
                                    if self.good_posture_start_time is None:
                                        self.good_posture_start_time = current_time
                                        print("Bắt đầu đếm tư thế TỐT")

                                    # reset timer xấu
                                    self.bad_posture_start_time = None

                                    good_time = current_time - self.good_posture_start_time

                                    if good_time >= 10:
                                        self.send_led_command('B')
                                    else:
                                        self.send_led_command('O')

                                # ======== TƯ THẾ XẤU ========
                                elif is_bad_posture:
                                    if self.bad_posture_start_time is None:
                                        self.bad_posture_start_time = current_time
                                        print("Bắt đầu đếm tư thế XẤU")

                                    # reset timer tốt
                                    self.good_posture_start_time = None

                                    bad_time = current_time - self.bad_posture_start_time
                                    print(f"Thời gian giữ tư thế xấu: {bad_time:.1f}s ({stable_posture})")

                                    # 3 cấp độ (đv giây)
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

                session_time = self.session_time
                if not self.log_records or self.log_records[-1][0] < session_time:
                    self.log_records.append((session_time, self.current_posture))

                time.sleep(0.03)

    def cleanup_history(self):
        """Dọn dẹp và sửa lỗi history nếu có"""
        cleaned_history = []
        for item in self.history:
            if (isinstance(item, (list, tuple)) and 
                len(item) == 2 and 
                isinstance(item[0], (int, float)) and 
                isinstance(item[1], str)):
                cleaned_history.append((item[0], item[1]))
        self.history = cleaned_history

    def remove_duplicate_history(self):
        """Loại bỏ các mục trùng lặp liên tiếp trong history"""
        if not self.history:
            return
    
        cleaned_history = []
        last_posture = None
    
        for item in self.history:
            if (isinstance(item, (list, tuple)) and len(item) == 2):
                timestamp, posture = item
            
                # Chỉ thêm nếu tư thế khác với tư thế trước đó
                if posture != last_posture:
                    cleaned_history.append((timestamp, posture))
                    last_posture = posture
                else:
                    continue
    
        self.history = cleaned_history

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
        
        # Cập nhật độ tin cậy
        confidence_pct = self.confidence * 100
        self.confidence_label.config(text=f"{confidence_pct:.1f}%")
        
        # Cập nhật màu nhãn tư thế
        if self.current_posture == "ngoi thang":
            self.posture_label.config(fg='#10b981')  # Xanh lá
        elif self.current_posture == "khong thay nguoi":
            self.posture_label.config(fg='#6b7280')  # Xám
        else:
            self.posture_label.config(fg='#ef4444')  # Đỏ
    
    def check_alerts(self, posture):
        """Kiểm tra và tạo cảnh báo"""
        current_time = datetime.now()

        # Xử lý riêng cho trạng thái không thấy người
        if posture == "khong thay nguoi":
            # Nếu chưa có thời gian bắt đầu mất người thì đặt
            if not hasattr(self, 'no_person_start_time'):
                self.no_person_start_time = current_time
            else:
                duration = (current_time - self.no_person_start_time).total_seconds()
                # Nếu mất liên tục >= 60s thì mới cảnh báo
                if duration >= 60:
                    if not self.last_alert_time or (current_time - self.last_alert_time).total_seconds() >= 60:
                        self.last_alert_time = current_time
                        alert_message = "Không phát hiện người \ntrong 1 phút"
                        self.recent_alerts.appendleft((current_time, alert_message))
                        self.alerts_count += 1
                        print(f"Đã thêm cảnh báo: {alert_message}")
                        self.root.after(0, self.update_alerts_display)
            return
        else:
            # Nếu thấy người trở lại, reset timer
            if hasattr(self, 'no_person_start_time'):
                del self.no_person_start_time

        # LẤY NGƯỠNG TỪ CÀI ĐẶT - ĐẢM BẢO ĐỒNG BỘ VỚI HISTORY
        sensitivity_text = self.sensitivity_var.get()
        if "50" in sensitivity_text:
            threshold = 0.50
        elif "90" in sensitivity_text:
            threshold = 0.90
        else:
            threshold = 0.70  # mặc định Trung bình

        # Bỏ qua nếu không đạt ngưỡng, trừ khi là "khong thay nguoi"
        if posture != "khong thay nguoi" and self.confidence < threshold:
            return  # Bỏ qua nếu không đạt ngưỡng

        # Chỉ cảnh báo nếu đã qua thời gian cooldown
        if self.last_alert_time and (current_time - self.last_alert_time).total_seconds() < self.alert_cooldown_seconds:
            return  # Bỏ qua nếu chưa đủ thời gian

        self.last_alert_time = current_time  # Cập nhật thời gian cảnh báo cuối

        # Tên cảnh báo tiếng Việt
        alert_names = {
                "guc dau": "Tư thế không tốt: Gục đầu",
                "nga nguoi": "Tư thế không tốt: Ngả người",
                "ngoi thang": "Tư thế tốt: Ngồi thẳng",
                "quay trai": "Tư thế không tốt: Quay trái",
                "quay phai": "Tư thế không tốt: Quay phải",
                "chong tay": "Tư thế không tốt: Chống tay",
        }

        alert_message = alert_names.get(posture, f"Tư thế: {posture}")

        # Thêm vào danh sách cảnh báo gần đây
        self.recent_alerts.appendleft((current_time, alert_message))
    
        # tăng số cảnh báo với tư thế xấu
        if posture != "ngoi thang":
            self.alerts_count += 1
            print(f"Đã thêm cảnh báo: {alert_message} (Độ tin cậy: {self.confidence:.1%}, Ngưỡng: {threshold:.0%})")

        # Cập nhật hiển thị cảnh báo
        self.root.after(0, self.update_alerts_display)
    
    def update_alerts_display(self):
        """Cập nhật panel cảnh báo mà không destroy toàn bộ widget"""
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
            if self.is_monitoring and hasattr(self, "session_start_datetime"):
                # Tính dựa trên datetime để nhất quán với báo cáo
                delta = datetime.now() - self.session_start_datetime
                self.session_time = int(delta.total_seconds())

                # Cập nhật thời gian tư thế tốt mỗi giây
                if self.current_posture == "ngoi thang":
                    self.good_posture_time += 1

                # Cập nhật hiển thị
                self.update_session_stats()

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
        
        # Tính lại thời gian tư thế tốt từ history
        good_time = 0
        if self.history and self.session_time > 0:
            for i in range(len(self.history) - 1):
                ts1, posture1 = self.history[i]
                ts2, _ = self.history[i + 1]
                if posture1 == "ngoi thang":
                    good_time += (ts2 - ts1)
            # Tính phần cuối cùng
            last_ts, last_posture = self.history[-1]
            if last_posture == "ngoi thang":
                good_time += (self.session_time - last_ts)

        # Cập nhật % tư thế tốt
        good_percentage = (good_time / self.session_time) * 100 if self.session_time > 0 else 0
        self.good_posture_label.config(text=f"{good_percentage:.0f}%")

        # Cập nhật số lượng cảnh báo
        self.alerts_count_label.config(text=str(self.alerts_count))

        # Lưu lại vào biến good_posture_time để xuất báo cáo khớp
        self.good_posture_time = good_time

    def export_report(self):
        """Xuất báo cáo"""
        self.cleanup_history()
        self.remove_duplicate_history()

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
                    f.write(f"Tư thế hiện tại: {self.current_posture}\n")
                    f.write(f"Độ tin cậy hiện tại: {self.confidence:.1%}\n")
            
                    f.write("\n" + "=" * 50 + "\n")
                
                    f.write("\nTHỐNG KÊ TƯ THẾ\n")
                    f.write("-" * 50 + "\n")
            
                    # Thống kê số lần xuất hiện của mỗi tư thế
                    if self.history:
                        posture_stats = {}
                        total_time_in_posture = {}
                
                        # Đếm số lần xuất hiện của mỗi tư thế
                        for timestamp, posture in self.history:
                            posture_stats[posture] = posture_stats.get(posture, 0) + 1
                
                        # Tính thời gian cho mỗi tư thế
                        for i in range(len(self.history) - 1):
                            current_ts, current_posture = self.history[i]
                            next_ts, _ = self.history[i + 1]
                            duration = next_ts - current_ts
                            total_time_in_posture[current_posture] = total_time_in_posture.get(current_posture, 0) + duration
                
                        # Tính thời gian cho tư thế cuối cùng
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
                
                        # ĐÁNH GIÁ HIỆU SUẤT
                        f.write("=" * 50 + "\n\n")
                        f.write("ĐÁNH GIÁ HIỆU SUẤT\n")
                        f.write("-" * 50 + "\n")
                        # Tính % thời gian tư thế tốt
                        good_posture_percentage_ui = (self.good_posture_time / self.session_time) * 100 if self.session_time > 0 else 0
                    
                        f.write(f"Thời gian tư thế tốt: {good_posture_percentage_ui:.1f}%\n")
                
                        # Tính điểm tập trung trung bình
                        focus_score_total = 0
                        for posture, time_spent in total_time_in_posture.items():
                            score = self.focus_scores.get(posture, 0)
                            focus_score_total += score * time_spent
                        avg_focus_score = focus_score_total / total_session_time if total_session_time > 0 else 0
                        f.write(f"Điểm tập trung trung bình: {avg_focus_score:.1f}%\n")
                        
                    else:
                        f.write("Không có dữ liệu lịch sử để thống kê.\n")
                
                    f.write("=" * 50 + "\n\n")

                    f.write("LỊCH SỬ THAY ĐỔI TƯ THẾ CHI TIẾT\n")
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
        """Xuất file log gồm thời gian giám sát và tư thế nhận dạng theo thời gian"""
        if not self.log_records:
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
            with open(file_path, "w", encoding="utf-8") as f:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write("===== FILE LOG GIÁM SÁT TƯ THẾ =====\n")
                start_dt = datetime.fromtimestamp(self.session_start_time)
                f.write(f"Thời gian bắt đầu: {start_dt.strftime('%d/%m/%Y %H:%M:%S')}\n")
                end_time = datetime.fromtimestamp(self.monitor_stop_time)
                f.write(f"Thời gian dừng: {end_time.strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"Thời gian xuất file log: {now_str}\n")
                f.write("=" * 50 + "\n\n")

                f.write("THỜI GIAN - TƯ THẾ NHẬN DẠNG\n")
                f.write("-------------------------------------\n")
                # Ghi từng dòng trong history
                max_time = int(self.monitor_stop_time - self.session_start_time)

                #max_time = int((self.session_end_datetime - self.session_start_datetime).total_seconds())

                for t, posture in self.log_records:
                    if t > max_time:
                        break  # Bỏ các dòng vượt quá thời gian dừng
                    timestamp = time.strftime("%H:%M:%S", time.gmtime(t))
                    f.write(f"{timestamp} - {posture}\n")

            messagebox.showinfo("Hoàn tất", f"Đã xuất file log thành công:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu file log:\n{e}")

    def browse_log_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files","*.txt")])
        if not file_path:
            return

        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            # Kiểm tra file có đúng định dạng log
            valid = False
            for line in lines:
                if "-" in line and any(pose in line for pose in self.focus_scores.keys()):
                    valid = True
                    break

            if not valid:
                messagebox.showerror("Lỗi", "File không hợp lệ, hãy chọn đúng file log!")
                return

            self.log_file_path = file_path
            
            # Hiện biểu đồ
            if hasattr(self, "analysis_placeholder"):
                self.analysis_placeholder.pack_forget()

            self.show_log_analysis()

        except Exception:
            messagebox.showerror("Lỗi", "File không hợp lệ hoặc bị hỏng")

    def show_log_analysis(self):
        if not self.log_file_path:
            return
        
        if self.is_monitoring:  # Chỉ hiển thị biểu đồ khi đã dừng giám sát
            return
        
        # Ẩn placeholder nếu tồn tại
        if hasattr(self, "analysis_placeholder") and self.analysis_placeholder.winfo_ismapped():
            self.analysis_placeholder.pack_forget()

        good = 0; total = 0; focus_sum = 0
        with open(self.log_file_path, encoding='utf-8') as f:
            for line in f:
                if "-" not in line or ":" not in line:
                    continue
                try:
                    _, posture = line.strip().split(" - ")
                except:
                    continue
                total += 1
                score = self.focus_scores.get(posture.strip(), 0)
                focus_sum += score
                if posture.strip() == "ngoi thang":
                    good += 1
        if total == 0:
            return
        good_pct = good / total * 100
        avg_focus = focus_sum / total

        if avg_focus >= 80:
            level = "Tập trung cao"
        elif avg_focus >= 50:
            level = "Tập trung tương đối"
        else:
            level = "Không tập trung"
        if hasattr(self, "analysis_label"):
            self.analysis_label.pack_forget()
        if self.pie_canvas:
            self.pie_canvas.get_tk_widget().destroy()
        fig = Figure(figsize=(2,2))
        ax = fig.add_subplot(111)
        ax.pie([good_pct, 100-good_pct], labels=["Tốt","Khác"], autopct="%1.0f%%")
        ax.set_title(level, fontsize=8)

        self.pie_canvas = FigureCanvasTkAgg(fig, master=self.analysis_frame)
        self.pie_canvas.get_tk_widget().pack(expand=True)

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
   - Gục đầu: Cần điều chỉnh
   - Ngả người: Cần điều chỉnh
   - Quay trái/phải: Cần điều chỉnh
   - Chống tay: Cần điều chỉnh
4. Cách tính điểm tập trung:
   - Ngồi thẳng: 100 %      - Gục đầu: 40 %
   - Ngả người: 60 %        - Quay trái/phải: 70 %   
   - Chống tay: 80 %        - Không thấy người: 0 %
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
    
def main():
    root = tk.Tk()
    
    def toggle_monitoring_key(event):
        app.toggle_monitoring()
    
    def exit_app_key(event):
        if app.is_monitoring:
            app.stop_monitoring()
        if SERIAL_AVAILABLE:
            ser.close()
        root.quit()
    
    app = PostureMonitoringGUI(root)
    
    root.bind('<F1>', toggle_monitoring_key)
    root.bind('<Control-q>', exit_app_key)
    root.bind('<Escape>', lambda e: app.stop_monitoring() if app.is_monitoring else None)
    
    def on_closing():
        if app.is_monitoring:
            app.stop_monitoring()
        if SERIAL_AVAILABLE:
            ser.close()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.focus_set()
    
    root.mainloop()

if __name__ == "__main__":
    main()