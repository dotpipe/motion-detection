# **Motion Detection and Object Tracking System**

[![Working](https://img.shields.io/badge/Status-Working-brightgreen)](https://github.com/your-repository)  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-yellowgreen)](https://opencv.org/)  
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

This project is a **real-time motion detection and object tracking system** that captures moving objects, processes frames dynamically, and identifies unique objects with minimal human intervention. It uses **OpenCV**, **background subtraction**, and **image similarity metrics (SSIM)** to track objects across frames and classify them as new or existing objects. The system is designed to work efficiently in real-time, with a focus on automation and accuracy.

---
# Advanced Real-Time Object Detection and Tracking System

## Overview
This system provides a sophisticated approach to real-time object detection, tracking, and classification using computer vision techniques. It stands out for its dynamic object persistence and intelligent merging capabilities.

## Key Components

### 1. Background Subtraction and Motion Detection
- Uses `cv2.createBackgroundSubtractorMOG2` for robust motion detection
- Implements adaptive history tracking (500 frames) with optimized shadow removal
- Features dynamic threshold adjustment for varying lighting conditions

### 2. Intelligent Object Merging
The system employs a unique three-tier merging strategy:
- **Color-based Merging**: Analyzes mean color values with configurable tolerance
- **Spatial Proximity**: Merges objects based on dynamic distance thresholds
- **Motion Coherence**: Groups objects with similar movement patterns

### 3. Object Persistence and Recognition
- Maintains a database of detected objects in `CROPPED_DIR`
- Uses Structural Similarity Index (SSIM) for robust object matching
- Implements duplicate detection and cleanup via `dupes.py`

### 4. Real-time Processing Pipeline
1. Frame capture and preprocessing
2. Background subtraction
3. Contour detection and filtering
4. Dynamic object merging
5. Object recognition and tracking
6. Visual feedback with bounding boxes and labels

## Key Advantages

### Enhanced Accuracy
- Multi-factor object detection reduces false positives
- Dynamic threshold adaptation for varying environments
- Intelligent merging prevents object fragmentation

### Efficient Processing
- Optimized contour operations
- Selective frame resizing for performance
- Smart caching of object data

### Robust Object Recognition
- SSIM-based matching for reliable object identification
- Persistence across frame sequences
- Automatic duplicate management

## Files Structure

### live3.py
- Main implementation with full feature set
- Handles object tracking, recognition, and persistence
- Implements the complete processing pipeline

### real.py
- Lightweight version focusing on core detection
- Useful for testing and development
- Contains basic merging algorithms

### dupes.py
- Manages duplicate object detection
- Maintains database consistency
- Optimizes storage usage

## Technical Specifications
- Minimum object size: 500 pixels
- Color tolerance: 30 units (RGB space)
- Distance threshold: 50 pixels
- SSIM threshold: 0.14
- Frame resolution: 640x480 (configurable)

## Advantages Over Traditional Systems
1. **Dynamic Object Handling**
   - Traditional systems often use fixed thresholds
   - This system adapts to changing conditions

2. **Intelligent Merging**
   - Most systems treat objects independently
   - Our approach considers spatial and color relationships

3. **Persistent Recognition**
   - Common systems only track frame-to-frame
   - This implementation maintains object identity across sessions

4. **Resource Efficiency**
   - Optimized processing pipeline
   - Selective update mechanisms
   - Smart caching strategies

## Use Cases
- Security monitoring
- Object counting and tracking
- Motion analysis
- Behavioral pattern recognition
- Industrial quality control

## Future Improvements
1. Deep learning integration for enhanced recognition
2. Multi-camera support
3. Real-time performance optimization
4. Extended object classification capabilities

