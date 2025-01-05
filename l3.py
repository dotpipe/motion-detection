import cv2
import numpy as np
import os
import time
from collections import deque
from skimage.metrics import structural_similarity as ssim

CROPPED_DIR = "cropped_objects/"
os.makedirs(CROPPED_DIR, exist_ok=True)

def compare_strips(img1, img2, axis, threshold=0.2):
    if axis == 'x':
        strip1 = img1[:, :8]
        strip2 = img2[:, :8]
    else:  # y-axis
        strip1 = img1[:8, :]
        strip2 = img2[:8, :]
    
    common_size = (100, 8) if axis == 'x' else (8, 100)
    strip1 = cv2.resize(strip1, common_size)
    strip2 = cv2.resize(strip2, common_size)
    
    gray1 = cv2.cvtColor(strip1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(strip2, cv2.COLOR_BGR2GRAY)
    
    min_dim = min(gray1.shape)
    win_size = min(7, min_dim - (min_dim % 2) + 1)
    
    score, _ = ssim(gray1, gray2, win_size=win_size, full=True)
    return score > threshold

def enhance_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Increase saturation
    s = cv2.add(s, 30)
    
    # Normalize value channel
    v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    
    # Merge channels
    hsv = cv2.merge([h, s, v])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhanced

def merge_contours_by_color_and_motion(frame, contours, color_tolerance, distance_threshold):
    merged_contours = []
    while contours:
        current_contour = contours.pop(0)
        x, y, w, h = cv2.boundingRect(current_contour)
        current_color = cv2.mean(frame[y:y+h, x:x+w])[:3]

        merged_contour = current_contour
        to_remove = []
        for i, other_contour in enumerate(contours):
            ox, oy, ow, oh = cv2.boundingRect(other_contour)
            other_color = cv2.mean(frame[oy:oy+oh, ox:ox+ow])[:3]

            color_distance = np.linalg.norm(np.array(current_color) - np.array(other_color))
            spatial_distance = np.linalg.norm(np.array([x+w/2, y+h/2]) - np.array([ox+ow/2, oy+oh/2]))

            if color_distance < color_tolerance and spatial_distance < distance_threshold:
                merged_contour = np.vstack((merged_contour, other_contour))
                to_remove.append(i)

        contours = [c for i, c in enumerate(contours) if i not in to_remove]
        merged_contours.append(cv2.convexHull(merged_contour))

    return merged_contours

def process_frame(frame, back_sub, min_area=500, color_tolerance=30, distance_threshold=50):
    fg_mask = back_sub.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    merged_contours = merge_contours_by_color_and_motion(frame, filtered_contours, color_tolerance, distance_threshold)
    
    detected_objects = []
    files = sorted(os.listdir(CROPPED_DIR), key=lambda x: x.startswith("object_"))
    
    for contour in merged_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cropped_object = frame[y:y+h, x:x+w]
        cropped_object = enhance_color(cropped_object)
        match_found = False
        matched_filename = ""
        for filename in files:
            file_path = os.path.join(CROPPED_DIR, filename)
            stored_img = cv2.imread(file_path)
            
            if compare_strips(cropped_object, stored_img, 'x') and compare_strips(cropped_object, stored_img, 'y'):
                color = (255, 0, 0) if filename.startswith('object_') else (0, 0, 255)
                match_found = True
                matched_filename = filename
                break
        
        if not match_found:
            color = (0, 255, 0)
            matched_filename = f"object_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(os.path.join(CROPPED_DIR, matched_filename), cropped_object)
        
        detected_objects.append((x, y, w, h, color, matched_filename))
    
    return detected_objects

import time

def main_video_processing():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)
    frame_buffer = deque(maxlen=10)
    red_annotations = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from the camera.")
            break

        frame = cv2.resize(frame, (640, 480))
        frame_buffer.append(frame)

        if len(frame_buffer) == 10:
            oldest_frame = frame_buffer[0]
            current_frame = frame_buffer[5]
            detected_objects = process_frame(oldest_frame, back_sub)
            
            display_frame = current_frame.copy()
            current_time = time.time()
            
            for x, y, w, h, color, filename in detected_objects:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                if not filename.startswith('object_'):
                    label = os.path.splitext(filename)[0]
                    if color == (0, 0, 255):  # Red annotation
                        red_annotations[(x, y, w, h, label)] = current_time
                    cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display persistent red annotations
            for (x, y, w, h, label), timestamp in list(red_annotations.items()):
                if current_time - timestamp < 3:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    red_annotations.pop((x, y, w, h, label))
            
            cv2.imshow("Real-Time Video Processing", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_video_processing()

def enhance_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Increase saturation
    s = cv2.add(s, 30)
    
    # Normalize value channel
    v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    
    # Merge channels
    hsv = cv2.merge([h, s, v])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhanced
