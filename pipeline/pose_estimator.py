import os
import sys
import cv2
import requests


API_URL = "https://api-inference.huggingface.co/models/google/movenet-singlepose-thunder"

# MoveNet returns 17 keypoints in this order
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def get_hf_token():
    """
    Return the HuggingFace token from Streamlit secrets if available,
    otherwise fall back to the HF_TOKEN environment variable.
    """
    try:
        import streamlit as st
        return st.secrets["HF_TOKEN"]
    except Exception:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN not found. Set it in .streamlit/secrets.toml or as an env variable."
            )
        return token


def frame_to_bytes(frame):
    """
    Encode a BGR numpy array to JPEG bytes for the API request.
    """
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Failed to encode frame as JPEG.")
    return buffer.tobytes()


def parse_keypoints(response_json):
    """
    Parse the HuggingFace MoveNet API response into a flat dict:
        { "nose": {"x": float, "y": float, "confidence": float}, ... }

    MoveNet returns keypoints as a list of dicts with keys:
        "label", "score", "x", "y"
    """
    keypoints = {}

    # Response is a list of keypoint dicts: [{"label": "nose", "score": 0.9, "x": 0.5, "y": 0.3}, ...]
    if isinstance(response_json, list):
        for kp in response_json:
            label = kp.get("label")
            if label:
                keypoints[label] = {
                    "x": kp.get("x", 0.0),
                    "y": kp.get("y", 0.0),
                    "confidence": kp.get("score", 0.0),
                }

    return keypoints


def get_keypoints(frame):
    """
    Send a single frame (numpy array) to the HuggingFace MoveNet API
    and return a dict of keypoints.

    Args:
        frame: BGR numpy array.

    Returns:
        Dict mapping joint name → {"x": float, "y": float, "confidence": float}.
        Coordinates are normalized 0–1 relative to image dimensions.
    """
    token = get_hf_token()
    image_bytes = frame_to_bytes(frame)
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.post(API_URL, headers=headers, data=image_bytes)

    if response.status_code != 200:
        raise RuntimeError(
            f"API request failed ({response.status_code}): {response.text}"
        )

    return parse_keypoints(response.json())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pose_estimator.py <image_or_video_path>")
        sys.exit(1)

    path = sys.argv[1]

    # Accept either a static image or a video (use first frame)
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

    print(f"Sending frame ({frame.shape}) to HuggingFace MoveNet API...")
    keypoints = get_keypoints(frame)

    if not keypoints:
        print("No keypoints returned. Check the API response format.")
    else:
        print(f"Got {len(keypoints)} keypoints:")
        for name, kp in keypoints.items():
            print(f"  {name:20s}  x={kp['x']:.3f}  y={kp['y']:.3f}  conf={kp['confidence']:.3f}")
