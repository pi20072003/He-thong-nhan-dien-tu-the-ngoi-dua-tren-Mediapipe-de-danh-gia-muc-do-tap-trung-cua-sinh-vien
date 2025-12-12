import serial
import time

ser = serial.Serial('COM3', 115200, timeout=1)
time.sleep(2)  # Đợi khởi tạo

# Test từng lệnh
commands = ['W','R', 'Y', 'B', 'O']
for cmd in commands:
    print(f"Gửi: {cmd}")
    ser.write(cmd.encode())
    time.sleep(2)

ser.close() 