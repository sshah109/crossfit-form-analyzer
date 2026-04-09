import sys
import cv2


def extract_frames(video_path, sample_interval=10):
    """
    Open a video file and return one frame every `sample_interval` frames.

    Args:
        video_path: Path to the video file.
        sample_interval: Keep every Nth frame (default: 10).

    Returns:
        List of numpy arrays (BGR frames).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % sample_interval == 0:
            frames.append(frame)
        frame_index += 1

    cap.release()
    return frames


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_frames.py <video_path>")
        sys.exit(1)

    path = sys.argv[1]
    frames = extract_frames(path)
    print(f"Extracted {len(frames)} frames from '{path}'")
    print(f"Frame shape: {frames[0].shape if frames else 'N/A'}")
