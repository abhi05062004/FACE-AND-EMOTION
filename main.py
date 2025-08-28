import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def detect_face_parts_mediapipe():
    # Setup MediaPipe Face Mesh
    # max_num_faces: Set to 1 for single face detection, can be increased
    # refine_landmarks: Improves landmark accuracy, especially around eyes and mouth
    # min_detection_confidence: Minimum confidence for face detection to be successful
    # min_tracking_confidence: Minimum confidence for face tracking to be successful
    print("[INFO] Initializing MediaPipe Face Mesh...")
    try:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        print("[INFO] MediaPipe Face Mesh initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Could not initialize MediaPipe Face Mesh: {e}")
        print("Please ensure MediaPipe is installed (pip install mediapipe).")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Failed to open webcam. Exiting...")
        return

    print("[INFO] Starting video stream for face parts identification...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame from webcam. Exiting...")
            break

        # Flip the frame horizontally for a mirrored display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB. MediaPipe expects RGB input.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find face landmarks
        results = face_mesh.process(rgb_frame)

        # Draw the face mesh annotations on the image.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get frame dimensions for converting normalized coordinates
                img_h, img_w, _ = frame.shape

                # --- Draw specific facial features using MediaPipe's predefined connections ---

                # Jawline/Face Oval (Blue)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1))
                # Add text label for Jawline/Face Outline (using landmark 152 for position)
                if len(face_landmarks.landmark) > 152:
                    x, y = int(face_landmarks.landmark[152].x * img_w), int(face_landmarks.landmark[152].y * img_h)
                    cv2.putText(frame, "Face Outline", (x - 50, y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Eyebrows (Cyan - Left and Right)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1))
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1))
                # Add text label for Eyebrows (using landmark 70 for left eyebrow position)
                if len(face_landmarks.landmark) > 70:
                    x, y = int(face_landmarks.landmark[70].x * img_w), int(face_landmarks.landmark[70].y * img_h)
                    cv2.putText(frame, "Eyebrows", (x - 30, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


                # Nose (Magenta)
                # FACEMESH_NOSE_OUTLINE includes the bridge and tip of the nose
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_NOSE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1))
                # Add text label for Nose (using landmark 4 for nose tip position)
                if len(face_landmarks.landmark) > 4:
                    x, y = int(face_landmarks.landmark[4].x * img_w), int(face_landmarks.landmark[4].y * img_h)
                    cv2.putText(frame, "Nose", (x - 15, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                # Eyes (Green - Left and Right)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
                # Add text label for Eyes (using landmark 145 for left eye position)
                if len(face_landmarks.landmark) > 145:
                    x, y = int(face_landmarks.landmark[145].x * img_w), int(face_landmarks.landmark[145].y * img_h)
                    cv2.putText(frame, "Eyes", (x - 15, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Mouth (Red - Lips)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))
                # Add text label for Mouth (using landmark 13 for mouth position)
                if len(face_landmarks.landmark) > 13:
                    x, y = int(face_landmarks.landmark[13].x * img_w), int(face_landmarks.landmark[13].y * img_h)
                    cv2.putText(frame, "Mouth", (x - 20, y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # You can also draw a bounding box around the entire face based on MediaPipe landmarks
                # Iterate through all landmarks to find min/max x,y
                x_coords = [landmark.x for landmark in face_landmarks.landmark]
                y_coords = [landmark.y for landmark in face_landmarks.landmark]

                # Convert normalized coordinates to pixel coordinates
                min_x, max_x = int(min(x_coords) * img_w), int(max(x_coords) * img_w)
                min_y, max_y = int(min(y_coords) * img_h), int(max(y_coords) * img_h)

                # Draw bounding box
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)


        # Display the frame
        cv2.imshow("Face Parts Identification (OpenCV & MediaPipe)", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_face_parts_mediapipe()

