import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:

    def __init__(self, model_path='hand_landmarker.task'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.IMAGE)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def track(self):
        """
        Starts the webcam, displays the feed, and yields a numpy array of shape (N_HANDS, 21, 3).
        The array contains (x, y, z) for each landmark.
        Empty hands are filled with zeros to maintain fixed shape if desired, 
        or we can yield only detected hands. 
        To ensure a consistent shape (2, 21, 3), we'll pad with zeros.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue

                # Flip for mirror view
                frame = cv2.flip(frame, 1)

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=rgb_frame)

                results = self.landmarker.detect(mp_image)

                if results.hand_landmarks:
                    # Prepare output array (2 hands, 21 landmarks, 3 coords (x,y,z))
                    output_landmarks = np.zeros((2, 21, 3), dtype=np.float32)

                    for i, hand_landmarks in enumerate(results.hand_landmarks):
                        if i >= 2: break
                        for j, landmark in enumerate(hand_landmarks):
                            output_landmarks[i, j] = [
                                landmark.x, landmark.y, landmark.z
                            ]

                            # Visualization
                            h, w, _ = frame.shape
                            cv2.circle(
                                frame,
                                (int(landmark.x * w), int(landmark.y * h)), 5,
                                (0, 255, 0), -1)

                    # Display the frame
                    cv2.imshow('Hand Tracking', frame)
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break

                    yield output_landmarks
                else:
                    # Even if no hands, still display the feed so it doesn't freeze
                    cv2.imshow('Hand Tracking', frame)
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def close(self):
        self.landmarker.close()
