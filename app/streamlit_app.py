import os
import sys
import tempfile

import cv2
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.extract_frames import extract_frames
from pipeline.pose_estimator import get_keypoints
from pipeline.form_rules import check_form


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_KP_CONFIDENCE = 0.3

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

JOINT_COLOR = (202, 133, 0)
BONE_COLOR  = (255, 255, 255)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LIFT IQ",
    page_icon="🏋️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Reset & base ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    overflow: hidden !important;
}

[data-testid="stMain"], .main .block-container {
    background-color: #000000 !important;
    padding: 1.2rem 2.5rem 0.5rem 2.5rem !important;
    max-width: 100% !important;
}

/* ── Header ── */
h1 {
    color: #0085CA !important;
    font-size: 2rem !important;
    font-weight: 900 !important;
    letter-spacing: 6px !important;
    text-transform: uppercase !important;
    margin-bottom: 0 !important;
    line-height: 1.1 !important;
}

.tagline {
    color: #555555;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 2px;
    margin-bottom: 0;
}

/* ── Divider ── */
hr {
    border-color: #1a1a1a !important;
    margin: 0.6rem 0 0.8rem 0 !important;
}

/* ── Column gap ── */
[data-testid="stHorizontalBlock"] {
    gap: 2rem !important;
    align-items: flex-start !important;
}

/* ── Left column border ── */
[data-testid="stHorizontalBlock"] > div:first-child {
    border-right: 1px solid #1a1a1a;
    padding-right: 1.5rem;
}

/* ── Section labels ── */
.section-label {
    color: #555555;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* ── File uploader — compact ── */
[data-testid="stFileUploader"] {
    background-color: #0d0d0d !important;
    border: 1.5px dashed #0085CA !important;
    border-radius: 6px !important;
}

[data-testid="stFileUploader"] label {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #0d0d0d !important;
    padding: 0.4rem !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    padding: 0.3rem 0 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #444444 !important;
    font-size: 0.75rem !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] label {
    color: #555555 !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}

[data-testid="stSelectbox"] > div > div {
    background-color: #0d0d0d !important;
    border: 1px solid #222222 !important;
    color: #ffffff !important;
    border-radius: 5px !important;
    font-size: 0.9rem !important;
}

/* ── Analyze button ── */
[data-testid="stButton"] > button {
    background-color: #0085CA !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 900 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    padding: 0.55rem 1.5rem !important;
    width: 100% !important;
    margin-top: 0.3rem !important;
}

[data-testid="stButton"] > button:hover {
    background-color: #006fa8 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #0085CA !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
}

/* ── Right column placeholder ── */
.placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 55vh;
    color: #222222;
    text-align: center;
}

.placeholder-icon {
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
    opacity: 0.4;
}

.placeholder-text {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #2a2a2a;
}

/* ── Image caption ── */
[data-testid="caption"] {
    color: #333333 !important;
    font-size: 0.7rem !important;
}

/* ── Fault cards ── */
.fault-card {
    background-color: #0f0f0f;
    border-left: 3px solid #CC3333;
    border-radius: 0 5px 5px 0;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
}

.fault-id {
    color: #444444;
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 3px;
}

.fault-description {
    color: #eeeeee;
    font-size: 0.88rem;
    font-weight: 700;
    margin-bottom: 4px;
    line-height: 1.3;
}

.fault-cue {
    color: #0085CA;
    font-size: 0.78rem;
    font-style: italic;
    line-height: 1.4;
}

/* ── Success card ── */
.success-card {
    background-color: #081508;
    border-left: 3px solid #22cc44;
    border-radius: 0 5px 5px 0;
    padding: 0.7rem 0.9rem;
    color: #55cc77 !important;
    font-weight: 700;
    font-size: 0.88rem;
}

/* ── Alert ── */
[data-testid="stAlert"] {
    background-color: #150000 !important;
    border: 1px solid #661111 !important;
    border-radius: 5px !important;
    color: #ffaaaa !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 0.8rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #000000; }
::-webkit-scrollbar-thumb { background: #1a1a1a; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def annotate_frame(frame, keypoints):
    out = frame.copy()
    h, w = frame.shape[:2]
    for joint_a, joint_b in SKELETON_CONNECTIONS:
        a = keypoints.get(joint_a)
        b = keypoints.get(joint_b)
        if (a and b
                and a.get("confidence", 0) >= MIN_KP_CONFIDENCE
                and b.get("confidence", 0) >= MIN_KP_CONFIDENCE):
            p1 = (int(a["x"] * w), int(a["y"] * h))
            p2 = (int(b["x"] * w), int(b["y"] * h))
            cv2.line(out, p1, p2, BONE_COLOR, 2, cv2.LINE_AA)
    for kp_data in keypoints.values():
        if kp_data.get("confidence", 0) >= MIN_KP_CONFIDENCE:
            cx = int(kp_data["x"] * w)
            cy = int(kp_data["y"] * h)
            cv2.circle(out, (cx, cy), 5, JOINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 5, BONE_COLOR,  1,  cv2.LINE_AA)
    return out


def run_pipeline(video_path, lift_type):
    try:
        frames = extract_frames(video_path)
    except ValueError as e:
        return None, [], str(e)

    if not frames:
        return None, [], "No frames could be extracted from the video."

    n = len(frames)
    indices = sorted(set([0, n // 2, n - 1]))
    selected_frames = [frames[i] for i in indices]

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
            "No pose detected. Make sure your full body is visible and the frame is well-lit."
        )

    faults = check_form(key_keypoints, lift_type, keypoints_sequence=all_keypoints)

    key_frame = selected_frames[mid_idx]
    annotated_bgr = annotate_frame(key_frame, key_keypoints)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return annotated_rgb, faults, None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

# ── Header ──
st.title("LIFT IQ")
st.markdown('<p class="tagline">Film your lift. Find out what\'s off. Fix it.</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Two-column layout — always rendered ──
left, right = st.columns([35, 65])

with left:
    uploaded_file = st.file_uploader(
        "UPLOAD VIDEO",
        type=["mp4", "mov", "avi"],
        help="5–30 second clip. Film from the side.",
        label_visibility="visible",
    )
    lift_label = st.selectbox(
        "LIFT TYPE",
        ["Deadlift", "Back Squat"],
        label_visibility="visible",
    )
    analyze_clicked = st.button("Analyze Form", type="primary", use_container_width=True)

    if analyze_clicked:
        if not uploaded_file:
            st.error("Upload a video first.")
        else:
            lift_type = "deadlift" if lift_label == "Deadlift" else "back_squat"
            suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                with st.spinner("Analyzing..."):
                    result = run_pipeline(tmp_path, lift_type)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            st.session_state["result"] = result

with right:
    if "result" not in st.session_state or st.session_state["result"] is None:
        st.markdown(
            """
            <div class="placeholder">
                <div class="placeholder-icon">🎯</div>
                <div class="placeholder-text">Your analysis will appear here</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        annotated_rgb, faults, error = st.session_state["result"]

        if error:
            st.error(error)
        else:
            # Small image — fixed 400px width so it doesn't dominate the column
            st.image(annotated_rgb, width=400, caption="Key frame · MediaPipe pose detection")

            st.markdown("&nbsp;", unsafe_allow_html=True)

            if faults:
                st.markdown(
                    f'<p class="section-label">{len(faults)} fault{"s" if len(faults) != 1 else ""} detected</p>',
                    unsafe_allow_html=True,
                )
                for fault in faults:
                    st.markdown(
                        f"""
                        <div class="fault-card">
                            <div class="fault-id">{fault['id']}</div>
                            <div class="fault-description">{fault['description']}</div>
                            <div class="fault-cue">&#x25B6;&nbsp; {fault['cue']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="success-card">✓ &nbsp; No faults detected — looking solid.</div>',
                    unsafe_allow_html=True,
                )
