from ultralytics import YOLO
import cv2
import numpy as np
import math

def process_live_video(fine_tuned, output_path):
    model = YOLO(fine_tuned)
    cap = cv2.VideoCapture(0)  # Open webcam (0 for default camera)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for result in results.obb.data.tolist():
            x_center, y_center, width, height, angle, score, class_id = result

            corners = cv2.boxPoints(((x_center, y_center), (width, height), angle * 180 / np.pi))
            corners = np.int0(corners)

            centroid_x = int(x_center)
            centroid_y = int(y_center)

            top_mid = midpoint(corners[0], corners[1])
            bottom_mid = midpoint(corners[2], corners[3])

            if results.names[int(class_id)] == "cylinder":
                angle_of_orientation = 0
            else:
                delta_x = bottom_mid[0] - top_mid[0]
                delta_y = bottom_mid[1] - top_mid[1]
                angle_of_orientation = math.degrees(math.atan2(delta_y, delta_x))

                if angle_of_orientation < 0:
                    angle_of_orientation += 180
                if angle_of_orientation > 180:
                    angle_of_orientation = 360 - angle_of_orientation

            # Print information to the terminal
            print("--------------------------------------------")
            if results.names[int(class_id)] == 'cylinder':
                print(f"Object_class: {results.names[int(class_id)]}")
                print(f"Orientation Angle: 0 degrees")
            else:
                print(f"Object_class: {results.names[int(class_id)]}")
                print(f"Orientation Angle: {angle_of_orientation:.2f} degrees")

            print(f"Frame_Height: {height:.2f}, Frame_Width: {width:.2f}")
            print(f"Coordinates: ({centroid_x}, {centroid_y})")
            print("--------------------------------------------")

           # draw_colored_bounding_box(frame, corners)
            cv2.circle(frame, (centroid_x, centroid_y), 3, (0, 0, 255), -1)
            cv2.line(frame, top_mid, bottom_mid, (128, 0, 128), 4)
            label = f'Angle: {angle_of_orientation:.2f} degrees'
            draw_text(frame, label, (centroid_x, centroid_y + 20), (255, 255, 255))

            class_label = f'{results.names[int(class_id)]}: {score:.2f}'
            draw_text_background(frame, class_label, (centroid_x, centroid_y - 5), (255, 0, 0))
            draw_text(frame, class_label, (centroid_x, centroid_y - 5), (255, 255, 255))

        # Draw bottom frame line
        line_color = (51, 87, 255)
        line_thickness = 5
        line_start = (0, frame.shape[0] - line_thickness)
        line_end = (frame.shape[1], frame.shape[0] - line_thickness)
        cv2.line(frame, line_start, line_end, line_color, line_thickness)

        midpoint_x = (line_start[0] + line_end[0]) // 2
        midpoint_y = (line_start[1] + line_end[1]) // 2

        cv2.circle(frame, (midpoint_x, midpoint_y), 3, (0, 255, 255), -1)
        cv2.putText(frame, "mpL", (midpoint_x, midpoint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Live Object Detection', frame)

        # Write the frame to the video file
        out.write(frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def draw_colored_bounding_box(frame, corners):
    cv2.line(frame, tuple(corners[0]), tuple(corners[1]), (0, 0, 255), 2)  # top-red
    cv2.line(frame, tuple(corners[1]), tuple(corners[2]), (255, 0, 0), 2)  # right-green
    cv2.line(frame, tuple(corners[2]), tuple(corners[3]), (0, 255, 0), 2)  # bottom-blue
    cv2.line(frame, tuple(corners[3]), tuple(corners[0]), (0, 255, 255), 2)  # left-yellow

def midpoint(p1, p2):
    return (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2

def draw_text_background(frame, text, position, color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(frame, (position[0] - 5, position[1] - text_size[1] - 10), (position[0] + text_size[0] + 5, position[1] + 5), color, cv2.FILLED)

def draw_text(frame, text, position, color):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# Example usage
fine_tuned = '/home/adithyadk/Desktop/checkerBoard/weights/train18-obb/best.pt'
output_path = '/home/adithyadk/Desktop/checkerBoard/OutputImages/Live_OBB_object_detection.mp4'

process_live_video(fine_tuned,output_path)




