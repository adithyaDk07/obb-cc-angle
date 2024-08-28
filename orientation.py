from ultralytics import YOLO
import cv2
import numpy as np
import os
import math

def process_video(input_path, output_path, model_file):
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load the trained model
    model = YOLO(model_file)

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get the dimensions and frame rate of the video
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Create the output video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Read the first frame of the video
    ret, frame = cap.read()

    while ret:
        # Perform object detection on the frame
        results = model(frame)[0]

        for result in results.obb.data.tolist():
            x_center, y_center, width, height, angle, score, class_id = result

            # Calculate the coordinates of the bounding box corners
            corners = cv2.boxPoints(((x_center, y_center), (width, height), angle * 180 / np.pi))
            corners = np.int0(corners)

            # Draw the bounding box with different colors for each edge
            draw_colored_bounding_box(frame, corners)

            # Calculate the centroid of the bounding box
            centroid_x = int(x_center)
            centroid_y = int(y_center)

            # Draw the centroid in red color
            cv2.circle(frame, (centroid_x, centroid_y), 3, (0, 0, 255), -1)

            # Calculate the midpoints of the top and bottom edges
            top_mid = midpoint(corners[0], corners[1])
            bottom_mid = midpoint(corners[2], corners[3])

            # Draw a line from the top midpoint to the bottom midpoint through the centroid
            cv2.line(frame, top_mid, bottom_mid, (128, 0, 128), 4)  # Purple line with increased thickness

            # Calculate the angle of orientation based on the reference protractor
            delta_x = bottom_mid[0] - top_mid[0]
            delta_y = bottom_mid[1] - top_mid[1]
            angle_of_orientation = math.degrees(math.atan2(delta_y, delta_x))

            # Adjust the angle to match the desired orientation (0 degrees on the right, 180 degrees on the left)
            if angle_of_orientation < 0:
                angle_of_orientation +=360
            if angle_of_orientation > 180:
                angle_of_orientation =360 - angle_of_orientation
                # Calculate the angle of orientation based on the protractor reference
    # Now, angle_of_orientation will range from 0 to 180 degrees, with 0 degrees on the right and 180 on the left

            # Transmit relevant data to the terminal
            object_name = results.names[int(class_id)]
            print(f"Object: {object_name}, Frame Width: {w}, Frame Height: {h}, "
                  f"Orientation Angle: {angle_of_orientation:.2f} degrees, "
                  f"Centroid: ({centroid_x}, {centroid_y})")

            # Draw the straight line at the bottom of the frame
        line_color = (51, 87, 255)  # Brown Red
        line_thickness = 5
        line_start = (0, h - line_thickness)
        line_end = (w, h - line_thickness)
        cv2.line(frame, line_start, line_end, line_color, line_thickness)

        midpoint_x = (line_start[0] + line_end[0]) // 2
        midpoint_y = (line_start[1] + line_end[1]) // 2

        # Draw the midpoint in yellow color
        cv2.circle(frame, (midpoint_x, midpoint_y), 3, (0, 255, 255), -1)

        # Display the text "mpL" at the midpoint in yellow color
        cv2.putText(frame, "mpL", (midpoint_x, midpoint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)
        ret, frame = cap.read()

    # Release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def draw_colored_bounding_box(frame, corners):
    # Draw the top edge in red
    cv2.line(frame, tuple(corners[0]), tuple(corners[1]), (0, 0, 255), 2)
    # Draw the right edge in blue
    cv2.line(frame, tuple(corners[1]), tuple(corners[2]), (255, 0, 0), 2)
    # Draw the bottom edge in green
    cv2.line(frame, tuple(corners[2]), tuple(corners[3]), (0, 255, 0), 2)
    # Draw the left edge in yellow
    cv2.line(frame, tuple(corners[3]), tuple(corners[0]), (0, 255, 255), 2)

def midpoint(p1, p2):
    return (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2

# Example usage
input_path = '/home/adithyadk/Desktop/checkerBoard/videos/12.mp4'
output_path = '/home/adithyadk/Desktop/checkerBoard/OutputImages/video_OBB_object_detection-orientation.mp4'
model_file = '/home/adithyadk/Desktop/checkerBoard/best (2).pt'

process_video(input_path, output_path, model_file)
