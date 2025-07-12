# detectors/feature_extractor.py
import cv2
import mediapipe as mp


def extract_pose_features(image, static_image_mode=True, min_detection_confidence=0.5):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=static_image_mode, min_detection_confidence=min_detection_confidence)

    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pose.close()

    if not results.pose_landmarks:
        return []

    landmarks = results.pose_landmarks.landmark
    feature = []
    for lm in landmarks:
        feature.extend([lm.x, lm.y])
    return feature