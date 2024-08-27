import cv2
from ultralytics import YOLO
import os
import numpy as np

def process_video(output_path, model_file):
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load the trained model
    model = YOLO(model_file)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Get the dimensions and frame rate of the webcam
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Create the output video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Perform object detection on the frame
        results = model(frame)[0]

        for result in results.obb.data.tolist():
            x_center, y_center, width, height, angle, score, class_id = result

            # Calculate the coordinates of the bounding box corners
            corners = cv2.boxPoints(((x_center, y_center), (width, height), angle * 180 / np.pi))
            corners = np.int0(corners)

            # Draw the bounding box
            draw_bounding_box(frame, corners, (255, 0, 0), 2)

            # Calculate the centroid of the bounding box
            centroid_x = int(x_center)
            centroid_y = int(y_center)

            # Draw the centroid in red color
            cv2.circle(frame, (centroid_x, centroid_y), 3, (0, 0, 255), -1)

            # Display class name and score
            label = f'{results.names[int(class_id)]}: {score:.2f}'

            # Draw a filled rectangle behind the text label
            draw_text_background(frame, label, (centroid_x, centroid_y - 5), (255, 87, 51))  # Specified color

            # Draw the text label
            draw_text(frame, label, (centroid_x, centroid_y - 5), (255, 255, 255))

        # Draw the straight line at the bottom of the frame
        line_color = (51, 87, 255)  # Brown Red
        line_thickness = 5
        line_start = (0, h - line_thickness)
        line_end = (w, h - line_thickness)
        cv2.line(frame, line_start, line_end, line_color, line_thickness)
        
        # Calculate the midpoint of the line
        midpoint_x = (line_start[0] + line_end[0]) // 2
        midpoint_y = (line_start[1] + line_end[1]) // 2
        
        # Draw the midpoint in yellow color
        cv2.circle(frame, (midpoint_x, midpoint_y), 7, (0, 255, 255), -1)
        
        # Display the text "mpL" at the midpoint in yellow color
        cv2.putText(frame, "mpL", (midpoint_x, midpoint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)

        # Show the processed frame
        cv2.imshow("Processed Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def draw_bounding_box(frame, corners, color, thickness):
    cv2.drawContours(frame, [corners], 0, color, thickness * 2)  # Increased thickness

def draw_text_background(frame, text, position, color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(frame, (position[0] - 5, position[1] - text_size[1] - 10), (position[0] + text_size[0] + 5, position[1] + 5), color, cv2.FILLED)  # Specified color

def draw_text(frame, text, position, color):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# Example usage
output_path = '/home/adithyadk/Desktop/checkerBoard/OutputImages/live_video_OBB_object_detection.mp4'
model_file = '/home/adithyadk/Desktop/checkerBoard/best (2).pt'

process_video(output_path, model_file)
