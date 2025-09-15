# Detectify: Real-Time Object Detection with OpenCV & YOLO

## 📌 Project Overview

This project demonstrates how to build a **real-time object detection
system** using **YOLOv5, OpenCV, and PyTorch**.\
The system can detect and classify multiple objects in both static
images and live video streams.

Object detection is applied in various fields such as: - 🚗 Self-driving
cars\
- 🎥 Video surveillance\
- 📱 Augmented reality\
- 🛒 Retail analytics

------------------------------------------------------------------------

## ⚙️ Project Workflow

1.  **Environment Setup**
    -   Install dependencies (`torch`, `torchvision`, `opencv-python`,
        `matplotlib`).
    -   Clone the YOLOv5 repository and set up requirements.
    -   Verify GPU/CPU availability.
2.  **Load Pre-Trained Model**
    -   Load YOLOv5 (small version `yolov5s`) pre-trained on the COCO
        dataset.
    -   Access COCO class labels (80 object categories).
3.  **Run Inference on Static Images**
    -   Test the model on example images.
    -   Visualize bounding boxes, labels, and confidence scores.
4.  **Real-Time Detection**
    -   Use OpenCV to capture video (from webcam or video file).
    -   Apply YOLOv5 inference on each frame.
    -   Display detected objects live.
5.  **Results & Visualization**
    -   Bounding boxes drawn around detected objects.
    -   Labels and confidence percentages displayed.

------------------------------------------------------------------------

## 🚀 Installation & Setup

Clone the repo and install dependencies:

``` bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

Install Python packages:

``` bash
pip install torch torchvision opencv-python matplotlib
```

------------------------------------------------------------------------

## ▶️ Usage

### Run on static images

``` python
# Load model
import torch
model = torch.hub.load('.', 'yolov5s', source='local', pretrained=True)

# Run inference on image
results = model("path/to/image.jpg")
results.show()
```

### Run real-time detection

``` python
import cv2

cap = cv2.VideoCapture(0)  # webcam
while True:
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow("YOLOv5 Detection", results.render()[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

------------------------------------------------------------------------

## 📂 Project Structure

-   `object_detection_project.ipynb` -- Jupyter notebook with full code\
-   `yolov5/` -- cloned YOLOv5 repository\
-   `requirements.txt` -- dependencies for YOLOv5

------------------------------------------------------------------------

## 🧾 Requirements

-   Python 3.7+
-   PyTorch
-   OpenCV
-   Matplotlib
-   YOLOv5 (Ultralytics repo)

------------------------------------------------------------------------

## ✨ Future Improvements

-   Train YOLOv5 on a **custom dataset**.\
-   Optimize for **edge devices** (e.g., Raspberry Pi, Jetson Nano).\
-   Deploy as a **web app or API**.
