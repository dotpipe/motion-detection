import os
import cv2
import numpy as np
import threading
import time
from skimage.metrics import structural_similarity as ssim

# Directory to save cropped bounding boxes
CROPPED_DIR = "cropped_objects/"
os.makedirs(CROPPED_DIR, exist_ok=True)  # Create directory if it doesn't exist

# Global flag for stopping threads
stop_threads = threading.Event()


# ==========================
# Duplicate Removal Functionality (from dupes.py)
# ==========================

def load_images_from_directory(directory):
    images = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Supported image formats
            image = cv2.imread(file_path)
            if image is not None:
                images[filename] = image
    return images


def are_images_similar(image1, image2, similarity_threshold=0.45):
    resized_image1 = cv2.resize(image1, (100, 100))
    resized_image2 = cv2.resize(image2, (100, 100))

    gray_image1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)

    similarity, _ = ssim(gray_image1, gray_image2, full=True)
    return similarity >= similarity_threshold


def remove_duplicates_in_real_time(directory, similarity_threshold=0.45, interval=1):
    """
    Continuously monitor the directory and remove duplicate images.
    """
    duplicates = set()
    while not stop_threads.is_set():
        images = load_images_from_directory(directory)
        # duplicates = set()
        filenames = list(images.keys())

        for i in range(len(filenames)):
            if filenames[i] in duplicates:
                continue

            for j in range(i + 1, len(filenames)):
                if filenames[j] in duplicates:
                    continue

                if are_images_similar(images[filenames[i]], images[filenames[j]], similarity_threshold):
                    print(f"Duplicate found: {filenames[j]} is similar to {filenames[i]}")
                    duplicates.add(filenames[j])

        # for duplicate in duplicates:
        #     duplicate_path = os.path.join(directory, duplicate)
    #        os.remove(duplicate_path)
    #        print(f"Removed duplicate: {duplicate}")

        time.sleep(interval)
    
    for duplicate in duplicates:
        duplicate_path = os.path.join(directory, duplicate)
        os.remove(duplicate_path)
        print(f"Removed duplicate: {duplicate}")


# ==========================
# Real-Time Video Processing (from real.py)
# ==========================

def get_mean_color(frame, box):
    x, y, w, h = box
    roi = frame[y:y + h, x:x + w]
    mean_color = cv2.mean(roi)[:3]
    return mean_color


def get_contour_center(contour):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return 0, 0
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def save_cropped_object(frame, contour):
    x, y, w, h = cv2.boundingRect(contour)
    cropped_object = frame[y:y + h, x:x + w]
    
    # Enhance contrast of the cropped object
    enhanced_object = enhance_contrast(cropped_object)
    
    # Generate a unique filename
    object_name = f"object_{int(time.time() * 1000)}.jpg"
    object_path = os.path.join(CROPPED_DIR, object_name)
    
    # Save the enhanced cropped object
    cv2.imwrite(object_path, enhanced_object)
    
    return object_name, (x, y, w, h)


def merge_contours_by_color_and_motion(frame, contours, color_tolerance, distance_threshold):
    merged_contours = []
    while contours:
        current_contour = contours.pop(0)
        x, y, w, h = cv2.boundingRect(current_contour)
        current_color = get_mean_color(frame, (x, y, w, h))

        merged_contour = current_contour
        to_remove_indices = []
        for i, other_contour in enumerate(contours):
            ox, oy, ow, oh = cv2.boundingRect(other_contour)
            other_color = get_mean_color(frame, (ox, oy, ow, oh))

            color_distance = np.linalg.norm(np.array(current_color) - np.array(other_color))
            cx1, cy1 = get_contour_center(current_contour)
            cx2, cy2 = get_contour_center(other_contour)
            distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

            if color_distance < color_tolerance and distance < distance_threshold:
                merged_contour = np.vstack((merged_contour, other_contour))
                to_remove_indices.append(i)

        contours = [contour for i, contour in enumerate(contours) if i not in to_remove_indices]
        merged_contour = cv2.convexHull(merged_contour)
        merged_contours.append(merged_contour)

    return merged_contours


def process_frame(frame, back_sub, min_area=500, color_tolerance=30, distance_threshold=50):
    """
    Process a single frame to detect, label, and group moving objects dynamically with background awareness.

    Args:
        frame (numpy.ndarray): The current video frame.
        back_sub (cv2.BackgroundSubtractor): Background subtractor object.
        min_area (int): Minimum area of detected objects to consider.
        color_tolerance (int): Tolerance for color similarity (Euclidean distance).
        distance_threshold (int): Maximum distance between contours to consider for merging.

    Returns:
        numpy.ndarray: The processed frame with merged contours drawn and labeled.
    """
    global object_id_counter, tracked_objects

    # Apply background subtraction
    fg_mask = back_sub.apply(frame)

    # Remove shadows and noise
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Find contours of moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    # Merge overlapping contours dynamically
    merged_contours = merge_contours_by_color_and_motion(frame, filtered_contours, color_tolerance, distance_threshold)

    # Track and label objects

    for contour in merged_contours:
        # Save the cropped object before drawing any bounding boxes
        object_name, (x, y, w, h) = save_cropped_object(frame, contour)

        # Check if this object matches any image in the cropped_objects directory
        matched_file = search_for_match_in_directory(cv2.imread(os.path.join(CROPPED_DIR, object_name)), CROPPED_DIR)

        if matched_file:
            # Object matches an existing image
            label = matched_file
            color = (0, 0, 255)  # Red for matched objects
            print(f"Matched with: {matched_file}")
        else:
            # New object
            label = object_name
            color = (0, 255, 0)  # Green for new objects
            print(f"New object saved as: {object_name}")

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# Add this near the top of the file with other global variables
files_to_delete = set()

def search_for_match_in_directory(image, directory, similarity_threshold=0.26):
    """
    Search for a matching image in the specified directory, skipping files marked for deletion.
    """
    if image.shape[0] < 7 or image.shape[1] < 7:
        print("Cropped image is too small for SSIM comparison. Skipping...")
        return None

    for filename in os.listdir(directory):
        if filename in files_to_delete:
            continue  # Skip files marked for deletion

        file_path = os.path.join(directory, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            stored_image = cv2.imread(file_path)

            if stored_image is not None:
                if stored_image.shape[0] < 7 or stored_image.shape[1] < 7:
                    print(f"Stored image {filename} is too small for SSIM comparison. Skipping...")
                    continue

                resized_image = cv2.resize(image, (100, 100))
                resized_stored = cv2.resize(stored_image, (100, 100))

                smaller_dimension = min(resized_image.shape[0], resized_image.shape[1])
                win_size = smaller_dimension if smaller_dimension % 2 == 1 else smaller_dimension - 1
                win_size = max(win_size, 3)

                try:
                    similarity, _ = ssim(resized_image, resized_stored, full=True, win_size=win_size, channel_axis=-1)
                    if similarity >= similarity_threshold:
                        print(f"Match found with {filename}, similarity: {similarity}")
                        return filename
                except ValueError as e:
                    print(f"Error comparing images with SSIM: {e}")

    return None

# Modify the remove_duplicates_in_real_time function to use the files_to_delete set
def remove_duplicates_in_real_time(directory, similarity_threshold=0.2, interval=14):
    global files_to_delete
    while not stop_threads.is_set():
        images = load_images_from_directory(directory)
        filenames = list(images.keys())

        for i in range(len(filenames)):
            if filenames[len(filenames) - 1 - i] in files_to_delete:
                continue

            for j in range(i + 1, len(filenames)):
                if filenames[len(filenames) - 1 - j] in files_to_delete:
                    continue

                if are_images_similar(images[filenames[len(filenames) - 1 - i]], images[filenames[len(filenames) - 1 - j]], similarity_threshold):
                    print(f"Duplicate found: {filenames[len(filenames) - 1 - i]} is similar to {filenames[j]}")
                    files_to_delete.add(filenames[len(filenames) - 1 - j])

        time.sleep(interval)
    
        for duplicate in files_to_delete:
            duplicate_path = os.path.join(directory, duplicate)
            if duplicate_path and os.path.exists(duplicate_path) and duplicate[:7] == "object_":
                os.remove(duplicate_path)
            print(f"Removed duplicate: {duplicate}")
        
        files_to_delete.clear()


def main_video_processing():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)
    print("Press 'q' to exit the real-time video processing.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from the camera.")
                break

            frame = cv2.resize(frame, (640, 480))
            processed_frame = process_frame(frame, back_sub)
            cv2.imshow("Real-Time Video Processing", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ==========================
# Main Entry Point
# ==========================

if __name__ == "__main__":
    try:
        # Start the duplicate-removal thread
        duplicate_thread = threading.Thread(target=remove_duplicates_in_real_time, args=(CROPPED_DIR, 0.15))
        duplicate_thread.daemon = True
        duplicate_thread.start()

        # Run the main video processing
        main_video_processing()
    except KeyboardInterrupt:
        print("Exiting program...")
    finally:
        stop_threads.set()
        # duplicate_thread.join()
        remove_duplicates_in_real_time(CROPPED_DIR, 0.15)
        print("Program terminated.")
