import tensorflow as tf
import mediapipe as mp
import google.protobuf as protobuf
import sklearn
import cv2
import pandas as pd
import numpy as np
import joblib
import struct
import seaborn as sns
import matplotlib as plt

print("TensorFlow:", tf.__version__)
print("Mediapipe:", mp.__version__)
print("Protobuf:", protobuf.__version__)
print("Scikit-learn:", sklearn.__version__)
print("OpenCV:", cv2.__version__)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Joblib:", joblib.__version__)
print("x",struct.calcsize("P") * 8)
print("Seaborn:", sns.__version__)
print("Matplotlib:", plt.__version__)