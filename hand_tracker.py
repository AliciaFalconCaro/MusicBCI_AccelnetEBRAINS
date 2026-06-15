import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():
    # Configure Hand Landmarker
    model_path = 'hand_landmarker.task'
    # Note: You need to download the model file from 
    # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models
    
    # For this example, if the model file is not present, we must inform the user
    import os
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please download it from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2
    )
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        # Initialize Webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Starting hand tracking (Tasks API)... Press 'q' to exit.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB before processing.
            # MediaPipe Tasks requires a mediapipe.Image object
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process the frame
            results = landmarker.detect(mp_image)

            # Draw hand landmarks
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    # Draw landmarks manually or use drawing utils if available
                    for landmark in hand_landmarks:
                        # Convert normalized coordinates to pixel coordinates
                        h, w, _ = frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                        
                        # Print wrist (index 0 is wrist)
                        # We need to find the wrist index
                    
                    # Print wrist coordinates (landmark 0)
                    wrist = hand_landmarks[0]
                    print(f"Wrist coordinates: x={wrist.x:.2f}, y={wrist.y:.2f}")

            # Display the resulting frame
            cv2.imshow('Hand Tracking', frame)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
