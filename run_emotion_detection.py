import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # Import load_model

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def run_emotion_detection_inference():
    # Path to your trained emotion model
    # IMPORTANT: Ensure 'emotion_model.h5' is in the same directory as this script,
    # or provide its full absolute path.
    # Corrected path format: using forward slashes instead of backslashes
    model_path = 'C:/Users/ABHIJITH/Documents/face detection/emotion_model.h5'

    # Define the emotion labels (MUST match the order used during training)
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    img_height, img_width = 48, 48 # Must match the input size used during training

    # --- Load the trained Keras model ---
    print(f"[INFO] Loading emotion model from: {model_path}...")
    try:
        model = load_model(model_path)
        print("[INFO] Emotion model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Could not load emotion model: {e}")
        print("Please ensure 'emotion_model.h5' exists and is a valid Keras model file.")
        return

    # --- Initialize MediaPipe Face Mesh ---
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

    # --- Initialize webcam ---
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Failed to open webcam. Exiting...")
        return

    print("[INFO] Starting video stream for real-time emotion detection...")

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

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get frame dimensions for converting normalized coordinates
                img_h, img_w, _ = frame.shape

                # --- Extract face ROI using MediaPipe landmarks ---
                x_coords = [landmark.x for landmark in face_landmarks.landmark]
                y_coords = [landmark.y for landmark in face_landmarks.landmark]

                min_x, max_x = int(min(x_coords) * img_w), int(max(x_coords) * img_w)
                min_y, max_y = int(min(y_coords) * img_h), int(max(y_coords) * img_h)

                # Add some padding to the bounding box for better face extraction
                padding = 20 # pixels
                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x = min(img_w - 1, max_x + padding)
                max_y = min(img_h - 1, max_y + padding)

                # Ensure bounding box is valid and extract face ROI
                if min_x < max_x and min_y < max_y:
                    face_roi = frame[min_y:max_y, min_x:max_x]

                    if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                        # Preprocess the face ROI for the emotion model
                        # Convert to grayscale
                        gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        # Resize to the model's input size (48x48)
                        resized_face = cv2.resize(gray_face_roi, (img_height, img_width))
                        # Normalize pixel values
                        normalized_face = resized_face / 255.0
                        # Add batch dimension and channel dimension (for grayscale, it's 1)
                        input_tensor = np.expand_dims(np.expand_dims(normalized_face, axis=0), axis=-1)

                        # --- Predict emotion using the loaded Keras model ---
                        emotion_preds = model.predict(input_tensor, verbose=0)[0] # verbose=0 to suppress prediction logs
                        predicted_emotion_idx = np.argmax(emotion_preds)
                        predicted_emotion = EMOTION_LABELS[predicted_emotion_idx]
                        emotion_confidence = emotion_preds[predicted_emotion_idx] * 100

                        # --- Draw results on the frame ---
                        # Draw bounding box around the face
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

                        # Display emotion label
                        emotion_text = f"{predicted_emotion}: {emotion_confidence:.2f}%"
                        cv2.putText(frame, emotion_text, (min_x, min_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        # You can also optionally draw face parts if desired,
                        # similar to the previous version of the code.
                        # For brevity, only emotion is shown here.

        # Display the frame
        cv2.imshow("Real-time Emotion Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close() # Close MediaPipe resources

if __name__ == "__main__":
    run_emotion_detection_inference()
