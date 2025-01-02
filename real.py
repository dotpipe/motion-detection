import cv2
import numpy as np


def process_frame(frame, back_sub, min_area=500, color_tolerance=30, distance_threshold=50):
    """
    Process a single frame to detect and group moving objects dynamically with background awareness.

    Args:
        frame (numpy.ndarray): The current video frame.
        back_sub (cv2.BackgroundSubtractor): Background subtractor object.
        min_area (int): Minimum area of detected objects to consider.
        color_tolerance (int): Tolerance for color similarity (Euclidean distance).
        distance_threshold (int): Maximum distance between contours to consider for merging.

    Returns:
        numpy.ndarray: The processed frame with merged contours drawn.
    """
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

    # Draw only the merged contours (green)
    for contour in merged_contours:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)  # Green for merged contours

    return frame


def merge_contours_by_color_and_motion(frame, contours, color_tolerance, distance_threshold):
    """
    Merge contours dynamically based on color and motion.

    Args:
        frame (numpy.ndarray): The current video frame.
        contours (list): List of contours to merge.
        color_tolerance (int): Tolerance for color similarity (Euclidean distance).
        distance_threshold (int): Maximum distance between contours to consider for merging.

    Returns:
        list: List of merged contours.
    """
    merged_contours = []
    while contours:
        # Take the first contour and its mean color
        current_contour = contours.pop(0)
        x, y, w, h = cv2.boundingRect(current_contour)
        current_color = get_mean_color(frame, (x, y, w, h))

        # Group contours with similar color and proximity
        merged_contour = current_contour
        to_remove_indices = []  # Store indices of contours to remove
        for i, other_contour in enumerate(contours):
            ox, oy, ow, oh = cv2.boundingRect(other_contour)
            other_color = get_mean_color(frame, (ox, oy, ow, oh))

            # Check color similarity
            color_distance = np.linalg.norm(np.array(current_color) - np.array(other_color))

            # Check spatial proximity
            cx1, cy1 = get_contour_center(current_contour)
            cx2, cy2 = get_contour_center(other_contour)
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

            # Merge if both conditions are met
            if color_distance < color_tolerance and distance < distance_threshold:
                merged_contour = np.vstack((merged_contour, other_contour))
                to_remove_indices.append(i)

        # Remove merged contours by indices
        contours = [contour for i, contour in enumerate(contours) if i not in to_remove_indices]

        # Use convex hull to clean up the merged contour
        merged_contour = cv2.convexHull(merged_contour)

        # Add the merged contour to the result
        merged_contours.append(merged_contour)

    return merged_contours


def get_mean_color(frame, box):
    """
    Calculate the mean color of the region inside a bounding box.

    Args:
        frame (numpy.ndarray): The current video frame.
        box (tuple): The bounding box (x, y, w, h).

    Returns:
        tuple: The mean color (B, G, R) of the region.
    """
    x, y, w, h = box
    roi = frame[y:y + h, x:x + w]  # Region of interest
    mean_color = cv2.mean(roi)[:3]  # Get mean color (B, G, R)
    return mean_color


def get_contour_center(contour):
    """
    Calculate the center of a contour.

    Args:
        contour (numpy.ndarray): The contour.

    Returns:
        tuple: The (x, y) center of the contour.
    """
    moments = cv2.moments(contour)
    if moments["m00"] == 0:  # Avoid division by zero
        return 0, 0
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def main():
    """
    Main function to run real-time video processing.
    """
    # Initialize video capture (0 = default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    # Create a background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)

    print("Press 'q' to exit the real-time video processing.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from the camera.")
            break

        # Resize frame for faster processing (optional)
        frame = cv2.resize(frame, (640, 480))

        # Process the frame
        processed_frame = process_frame(frame, back_sub)

        # Display the processed frame
        cv2.imshow("Real-Time Video Processing", processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
