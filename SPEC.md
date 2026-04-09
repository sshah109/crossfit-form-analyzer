# CrossFit Form Analyzer — Project Spec

## What This App Does

Users upload a short video (5–30 seconds) of themselves performing a barbell lift.
The app analyzes their movement frame by frame, detects their joint positions,
checks those positions against known good-form rules, and returns:

- An annotated image showing their joint positions and angles
- A plain-language list of form faults detected (e.g. "knees caving inward")
- A brief coaching cue for each fault

## Supported Lifts (v1 Prototype)

- Conventional deadlift
- Back squat

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| UI + hosting | Streamlit + Streamlit Community Cloud | Free, Python-native |
| Pose estimation | ViTPose via HuggingFace Inference API | Pre-trained, free tier |
| Video processing | OpenCV (cv2) | Frame extraction + annotation drawing |
| Form logic | Custom Python rule engine | Joint angle thresholds per lift |
| Secrets management | Streamlit secrets (`.streamlit/secrets.toml`) | Keeps API keys out of code |
| Version control | GitHub | Source of truth for all code |

## Folder Structure

```
crossfit-form-analyzer/
├── app/
│   └── streamlit_app.py         # Main Streamlit UI
├── pipeline/
│   ├── extract_frames.py        # Video → keyframes using OpenCV
│   ├── pose_estimator.py        # Keyframes → joint keypoints via HuggingFace
│   └── form_rules.py            # Keypoints → detected faults per lift type
├── data/
│   └── keypoints/               # Saved keypoint sequences (for future ML training)
├── model/
│   └── train_classifier.py      # Future: LSTM classifier on keypoint sequences
├── tests/
│   └── test_pipeline.py         # Basic unit tests
├── .streamlit/
│   └── secrets.toml             # API keys — never commit this file
├── .gitignore                   # Excludes secrets, data files, venv
├── requirements.txt             # All Python dependencies
├── README.md                    # Project overview
└── SPEC.md                      # This file
```

## Pipeline — How It Works Step by Step

```
User uploads video
        ↓
extract_frames.py
  → Sample 1 frame every 10 frames using OpenCV
  → Return list of frame arrays
        ↓
pose_estimator.py
  → Send best keyframe to HuggingFace ViTPose API
  → Receive keypoints: {nose, left_knee, right_knee, left_hip, right_hip,
                        left_shoulder, right_shoulder, left_ankle, right_ankle}
  → Each keypoint = {x, y, confidence_score}
        ↓
form_rules.py
  → Calculate joint angles from keypoint coordinates
  → Run lift-specific rule checks (see Form Rules section below)
  → Return list of detected faults with coaching cues
        ↓
streamlit_app.py
  → Draw skeleton + angle annotations on keyframe using OpenCV
  → Display annotated image
  → Display fault list with coaching cues
```

## Form Rules — v1 (Rule-Based, No ML Yet)

### Deadlift Faults

| Fault ID | What We're Checking | How We Detect It |
|---|---|---|
| `DL_ROUNDED_BACK` | Lower back rounding (spine not neutral) | Hip-shoulder-head angle deviates > 15° from straight |
| `DL_HIPS_RISE_FIRST` | Butt rises before chest (stripper deadlift) | Hip keypoint rises faster than shoulder keypoint across frames |
| `DL_BAR_DRIFT` | Bar drifts away from body | Horizontal distance between wrist and midfoot > threshold |
| `DL_SQUAT_SETUP` | Hips set too low (treating it like a squat) | Hip angle at setup < 45° (should be a hinge, not a squat) |

### Back Squat Faults

| Fault ID | What We're Checking | How We Detect It |
|---|---|---|
| `SQ_KNEE_CAVE` | Knees caving inward (valgus collapse) | Knee x-coordinate moves inside ankle x-coordinate |
| `SQ_DEPTH` | Not hitting depth (hip crease not below knee) | Hip y-coordinate does not exceed knee y-coordinate at bottom |
| `SQ_FORWARD_LEAN` | Excessive forward torso lean | Shoulder-hip vertical angle > 45° at bottom of squat |
| `SQ_HEEL_RISE` | Heels rising off floor | Ankle y-coordinate shifts significantly during descent |

## HuggingFace API Integration

Model: `google/movenet-singlepose-thunder`
Endpoint: `https://api-inference.huggingface.co/models/google/movenet-singlepose-thunder`
Auth: Bearer token stored in `.streamlit/secrets.toml` as `HF_TOKEN`

```python
# How to call the API (reference — do not hardcode token)
headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
response = requests.post(API_URL, headers=headers, data=image_bytes)
keypoints = response.json()
```

Rate limit on free tier: ~1,000 requests/day — sufficient for prototype use.

## Secrets and API Keys

Never hardcode tokens in Python files. Never commit `.streamlit/secrets.toml` to GitHub.

Local development:
```
.streamlit/secrets.toml  ← create this file manually, never push it
```

Contents of secrets.toml:
```toml
HF_TOKEN = "hf_your_token_here"
```

For Streamlit Cloud deployment: add secrets via the Streamlit Cloud dashboard UI
(Settings → Secrets), not via file.

## Build Order (Do This Sequence)

1. `pipeline/extract_frames.py` — build + test standalone with a sample video
2. `pipeline/pose_estimator.py` — build + test with a static image
3. `pipeline/form_rules.py` — build + test with hardcoded keypoint values
4. `app/streamlit_app.py` — wire all three together into the UI
5. Deploy to Streamlit Community Cloud

## v2 Roadmap (After Prototype Works)

- Collect labeled reps from CrossFit HQ YouTube videos using `yt-dlp`
- Extract keypoint sequences and label as good/bad form
- Train LSTM classifier in `model/train_classifier.py` using Google Colab (free GPU)
- Replace rule engine with ML classifier for more nuanced feedback
- Add Olympic lifts: clean, snatch, overhead squat

## Constraints

- All services must remain free (no paid API tiers)
- No user data is stored — videos are processed in memory and discarded
- App must run on Streamlit Community Cloud free tier (1GB RAM limit)
- Video uploads capped at 200MB (Streamlit's default limit)

## Key Decisions and Reasons

- **Why not retrain ViTPose?** Pose estimation works well out of the box.
  The differentiation is in the form rules, not the joint detection.
- **Why rule-based first?** Faster to prototype, easier to debug, establishes
  labeled data for future ML training.
- **Why HuggingFace Inference API vs local model?** Avoids memory issues on
  Streamlit's free tier. Model stays in HuggingFace's infrastructure.
- **Why OpenCV for annotation?** Lightweight, no extra dependencies, integrates
  directly with Streamlit's `st.image()`.
