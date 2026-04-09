import os
import sys
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ---------------------------------------------------------------------------
# Model — downloaded once to pipeline/ on first use
# ---------------------------------------------------------------------------

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_full.task")


# ---------------------------------------------------------------------------
# MediaPipe landmark index → our keypoint name
# ---------------------------------------------------------------------------

LANDMARK_MAP = {
    0:  "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

_landmarker = None


def _ensure_model():
    """Download the pose landmarker model file if not already on disk."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose model ({MODEL_URL})...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")


def _get_landmarker():
    """Lazy-initialise the PoseLandmarker (once per process)."""
    global _landmarker
    if _landmarker is None:
        _ensure_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    return _landmarker


def get_keypoints(frame):
    """
    Run MediaPipe Pose Landmarker on a single BGR frame and return keypoints.

    Args:
        frame: BGR numpy array.

    Returns:
        Dict mapping joint name → {"x": float, "y": float, "confidence": float}.
        Coordinates are normalized 0–1. Returns empty dict if no person detected.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    results = _get_landmarker().detect(mp_image)

    if not results.pose_landmarks:
        return {}

    landmarks = results.pose_landmarks[0]  # single-person mode

    keypoints = {}
    for idx, name in LANDMARK_MAP.items():
        lm = landmarks[idx]
        keypoints[name] = {
            "x":          float(lm.x),
            "y":          float(lm.y),
            "confidence": float(lm.visibility) if lm.visibility is not None else 1.0,
        }

    return keypoints


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pose_estimator.py <image_or_video_path>")
        sys.exit(1)

    path = sys.argv[1]

    if path.lower().endswith((".jpg", ".jpeg", ".png")):
        frame = cv2.imread(path)
        if frame is None:
            print(f"Could not read image: {path}")
            sys.exit(1)
    else:
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Could not read first frame from video: {path}")
            sys.exit(1)

    print(f"Running MediaPipe Pose Landmarker on frame {frame.shape}...")
    keypoints = get_keypoints(frame)

    if not keypoints:
        print("No pose detected. Try a frame where the full body is visible.")
    else:
        print(f"Got {len(keypoints)} keypoints:")
        for name, kp in keypoints.items():
            print(f"  {name:20s}  x={kp['x']:.3f}  y={kp['y']:.3f}  conf={kp['confidence']:.3f}")
