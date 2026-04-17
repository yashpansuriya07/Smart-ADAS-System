# ADAS System using Computer Vision & YOLO
Overview

This project implements a basic Advanced Driver Assistance System (ADAS) using computer vision techniques and deep learning. It performs:

Lane Detection
Vehicle Detection (YOLOv8)
Collision Warning System

The system processes a driving video and overlays real-time insights to simulate intelligent driving assistance.

🚀 Features

🛣️ Lane Detection
Region of Interest (ROI) masking,
Edge detection using Canny,
Lane extraction using Hough Transform,
Slope filtering and lane smoothing.

🚘 Object Detection
Uses YOLOv8,
Detects vehicles such as:
Cars,
Bikes,
Buses.

⚠️ Collision Warning
Detects vehicles in front lane,
Uses:
Bounding box area,
Object height,
Distance from frame center.

COLLISION WARNING!

🧠 How It Works:
Convert frame → grayscale,
Apply Gaussian Blur (reduce noise),
Detect edges using Canny,
Mask Region of Interest (road area),
Detect lane lines using Hough Transform,
Run YOLOv8 for object detection,
Apply collision logic based on:
Object size,
Position,
Lane alignment.

🛠️ Tech Stack
Python,
OpenCV,
NumPy,
Ultralytics YOLOv8.
