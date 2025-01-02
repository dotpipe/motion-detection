# **Motion Detection and Object Tracking System**

[![Working](https://img.shields.io/badge/Status-Working-brightgreen)](https://github.com/your-repository)  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-yellowgreen)](https://opencv.org/)  
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

This project is a **real-time motion detection and object tracking system** that captures moving objects, processes frames dynamically, and identifies unique objects with minimal human intervention. It uses **OpenCV**, **background subtraction**, and **image similarity metrics (SSIM)** to track objects across frames and classify them as new or existing objects. The system is designed to work efficiently in real-time, with a focus on automation and accuracy.

---

## **Table of Contents**
- [Features](#features)
- [Use Cases](#use-cases)
- [Installation](#installation)
- [How It Works](#how-it-works)
  - [real.py](#realpy)
  - [dupes.py](#dupespy)
  - [live3.py](#live3py)
- [Red-Green Attribute](#red-green-attribute)
- [Pleated Buffering Mechanism](#pleated-buffering-mechanism)
- [Directory Structure](#directory-structure)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## **Features**
- **Real-Time Motion Detection**: Detects and tracks moving objects in video frames using background subtraction.
- **Object Cropping and Classification**: Captures bounding boxes of moving objects and classifies them as new or duplicate using **SSIM (Structural Similarity Index)**.
- **Automated Object Tracking**: Tracks objects across frames with minimal human intervention.
- **Threaded Processing**: Uses threading to handle real-time video processing efficiently.
- **Pleated Buffering Mechanism**: Implements a `[prev][--][next]` frame buffering mechanism to improve object detection and tracking accuracy.
- **Red-Green Attribute**: Highlights matched objects in red and new objects in green for easy visualization.
- **Minimal Human Interaction**: Designed to work autonomously with minimal user input.

---

## **Use Cases**
1. **Surveillance Systems**: Automatically detect and track intruders or moving objects in a monitored area.
2. **Inventory Management**: Identify and classify new or duplicate items in a warehouse or storage facility.
3. **Traffic Monitoring**: Track vehicles and classify them as new or recurring based on their movement.
4. **Wildlife Observation**: Detect and track animals in a natural habitat with minimal human interference.
5. **Educational Projects**: Serve as a foundation for students learning about computer vision, motion detection, and real-time processing.

---

## **Installation**

### **Prerequisites**
1. Python 3.8 or higher
2. OpenCV 4.5 or higher
3. Required Python libraries:
   - `numpy`
   - `opencv-python`
   - `scikit-image`

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/motion-detection.git
   cd motion-detection
