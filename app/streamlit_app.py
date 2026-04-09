import os
import sys
import tempfile

import cv2
import streamlit as st

# Allow imports from project root regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.extract_frames import extract_frames
from pipeline.pose_estimator import get_keypoints
from pipeline.form_rules import check_form


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_KP_CONFIDENCE = 0.3

# Skeleton edges to draw between joints
SKELETON_CONNECTIONS = [
    ("nose",           "left_shoulder"),
    ("nose",           "right_shoulder"),
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
]

JOINT_COLOR = (202, 133, 0)    # CrossFit blue in BGR (#0085CA → B=202, G=133, R=0)
BONE_COLOR  = (255, 255, 255)  # white


# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CrossFit Form Analyzer",
    page_icon="🏋️",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Theme — dark CrossFit aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Global ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #000000 !important;
    color: #ffffff !important;
}

[data-testid="stMain"], .main .block-container {
    background-color: #000000 !important;
    padding-top: 2rem;
    max-width: 860px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111111 !important;
}

/* ── Typography ── */
h1 {
    color: #0085CA !important;
    font-size: 2.4rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.5px;
    text-transform: uppercase;
    margin-bottom: 0.2rem !important;
}

h2, h3 {
    color: #ffffff !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

p, li, label, .stMarkdown {
    color: #e0e0e0 !important;
}

/* ── Subheader accent line ── */
h2::after {
    content: "";
    display: block;
    width: 48px;
    height: 3px;
    background: #0085CA;
    margin-top: 6px;
    border-radius: 2px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background-color: #1a1a1a !important;
    border: 2px dashed #0085CA !important;
    border-radius: 8px !important;
    padding: 1rem;
}

[data-testid="stFileUploader"] label {
    color: #ffffff !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #1a1a1a !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #aaaaaa !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] label {
    color: #ffffff !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
}

[data-testid="stSelectbox"] > div > div {
    background-color: #1a1a1a !important;
    border: 1px solid #333333 !important;
    color: #ffffff !important;
    border-radius: 6px !important;
}

/* ── Primary button ── */
[data-testid="stButton"] > button[kind="primary"],
[data-testid="stButton"] > button {
    background-color: #0085CA !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 900 !important;
    font-size: 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    padding: 0.6rem 2rem !important;
    transition: background-color 0.15s ease;
}

[data-testid="stButton"] > button:hover {
    background-color: #006fa8 !important;
}

[data-testid="stButton"] > button:disabled {
    background-color: #333333 !important;
    color: #666666 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #0085CA !important;
    font-weight: 700 !important;
}

/* ── Image caption ── */
[data-testid="caption"] {
    color: #888888 !important;
    font-size: 0.8rem !important;
}

/* ── Fault cards ── */
.fault-card {
    background-color: #1a1a1a;
    border-left: 4px solid #CC3333;
    border-radius: 0 6px 6px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}

.fault-id {
    color: #888888;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

.fault-description {
    color: #ffffff;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 6px;
}

.fault-cue {
    color: #0085CA;
    font-size: 0.9rem;
    font-style: italic;
}

/* ── Success card ── */
.success-card {
    background-color: #0d2d0d;
    border-left: 4px solid #22cc44;
    border-radius: 0 6px 6px 0;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #88ee88 !important;
    font-weight: 700;
    font-size: 1rem;
}

/* ── Error card ── */
[data-testid="stAlert"] {
    background-color: #2a0000 !important;
    border: 1px solid #CC3333 !important;
    border-radius: 6px !important;
    color: #ffaaaa !important;
}

/* ── Divider ── */
hr {
    border-color: #222222 !important;
    margin: 1.5rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #111111; }
::-webkit-scrollbar-thumb { background: #333333; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def annotate_frame(frame, keypoints):
    """Draw skeleton and joint circles onto a BGR frame. Returns BGR array."""
    out = frame.copy()
    h, w = frame.shape[:2]

    # Bones
    for joint_a, joint_b in SKELETON_CONNECTIONS:
        a = keypoints.get(joint_a)
        b = keypoints.get(joint_b)
        if (a and b
                and a.get("confidence", 0) >= MIN_KP_CONFIDENCE
                and b.get("confidence", 0) >= MIN_KP_CONFIDENCE):
            p1 = (int(a["x"] * w), int(a["y"] * h))
            p2 = (int(b["x"] * w), int(b["y"] * h))
            cv2.line(out, p1, p2, BONE_COLOR, 2, cv2.LINE_AA)

    # Joints
    for kp_data in keypoints.values():
        if kp_data.get("confidence", 0) >= MIN_KP_CONFIDENCE:
            cx = int(kp_data["x"] * w)
            cy = int(kp_data["y"] * h)
            cv2.circle(out, (cx, cy), 6, JOINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 6, BONE_COLOR,  1,  cv2.LINE_AA)

    return out


def run_pipeline(video_path, lift_type):
    """
    Extract frames → get keypoints → check form → annotate key frame.

    Returns:
        annotated_rgb: numpy array (RGB) ready for st.image, or None on error.
        faults:        list of fault dicts.
        error:         error string, or None on success.
    """
    # 1. Extract sampled frames
    try:
        frames = extract_frames(video_path)
    except ValueError as e:
        return None, [], str(e)

    if not frames:
        return None, [], "No frames could be extracted from the video."

    # 2. Choose up to 3 frames: first, middle, last (for temporal checks)
    n = len(frames)
    indices = sorted(set([0, n // 2, n - 1]))
    selected_frames = [frames[i] for i in indices]

    # 3. Get keypoints for each selected frame
    all_keypoints = []
    for frame in selected_frames:
        try:
            kps = get_keypoints(frame)
            all_keypoints.append(kps)
        except RuntimeError as e:
            return None, [], f"Pose estimation failed: {e}"

    mid_idx = len(all_keypoints) // 2
    key_keypoints = all_keypoints[mid_idx]

    if not key_keypoints:
        return None, [], (
            "The pose estimator returned no keypoints. "
            "The model may still be loading on HuggingFace — wait 20 seconds and try again."
        )

    # 4. Check form rules
    faults = check_form(key_keypoints, lift_type, keypoints_sequence=all_keypoints)

    # 5. Annotate the middle frame and convert BGR → RGB for Streamlit
    key_frame = selected_frames[mid_idx]
    annotated_bgr = annotate_frame(key_frame, key_keypoints)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return annotated_rgb, faults, None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("CrossFit Form Analyzer")
st.markdown(
    "Upload a short video of your lift (5–30 sec). "
    "The app detects your joint positions and checks them against known good-form standards — "
    "then gives you plain-English feedback and coaching cues."
)

st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Upload your lift video",
        type=["mp4", "mov", "avi"],
        help="5–30 second clip works best. Film from the side for deadlifts and squats.",
    )
with col2:
    lift_label = st.selectbox(
        "Lift type",
        ["Deadlift", "Back Squat"],
    )

analyze_clicked = st.button("Analyze Form", type="primary", use_container_width=True)

if analyze_clicked:
    if not uploaded_file:
        st.error("Please upload a video first.")
    else:
        lift_type = "deadlift" if lift_label == "Deadlift" else "back_squat"
        suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()

        # Write upload to a temp file so OpenCV can read it
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner("Analyzing your form..."):
                annotated_rgb, faults, error = run_pipeline(tmp_path, lift_type)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # ── Results ──
        st.markdown("---")
        st.subheader("Form Analysis")

        if error:
            st.error(error)
        else:
            st.image(
                annotated_rgb,
                caption="Key frame — joints detected by MoveNet",
                use_container_width=True,
            )

            st.markdown("&nbsp;", unsafe_allow_html=True)

            if faults:
                st.subheader(f"{len(faults)} Fault{'s' if len(faults) != 1 else ''} Detected")
                for fault in faults:
                    st.markdown(
                        f"""
                        <div class="fault-card">
                            <div class="fault-id">{fault['id']}</div>
                            <div class="fault-description">{fault['description']}</div>
                            <div class="fault-cue">&#x25B6; {fault['cue']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    """
                    <div class="success-card">
                        ✓ &nbsp; No form faults detected — looking solid.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
