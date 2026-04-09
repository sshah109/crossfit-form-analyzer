import sys
import numpy as np


MIN_CONFIDENCE = 0.3

# How far a wrist can stray from the ankle (midfoot proxy) before flagging bar drift.
# Normalized coords: 0.1 ≈ 10% of frame width.
BAR_DRIFT_THRESHOLD = 0.10

# Ankle y-shift across frames that signals heel rise (normalized coords).
HEEL_RISE_THRESHOLD = 0.03


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def angle_at_joint(a, b, c):
    """
    Angle in degrees at point b, in the triangle a → b → c.
    Returns None if any point is missing.
    """
    if a is None or b is None or c is None:
        return None
    ba = np.array([a["x"] - b["x"], a["y"] - b["y"]])
    bc = np.array([c["x"] - b["x"], c["y"] - b["y"]])
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return None
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def vertical_angle(p1, p2):
    """
    Angle in degrees between the segment p1 → p2 and the vertical axis.
    0° = perfectly vertical, 90° = horizontal.
    Returns None if either point is missing.
    """
    if p1 is None or p2 is None:
        return None
    dx = p2["x"] - p1["x"]
    dy = p2["y"] - p1["y"]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))


def kp(keypoints, name):
    """
    Return a keypoint dict if it exists and meets the confidence threshold,
    otherwise return None.
    """
    point = keypoints.get(name)
    if point and point.get("confidence", 0) >= MIN_CONFIDENCE:
        return point
    return None


def avg_x(p1, p2):
    """Average x of two points, ignoring None."""
    pts = [p for p in [p1, p2] if p is not None]
    return sum(p["x"] for p in pts) / len(pts) if pts else None


def avg_y(p1, p2):
    """Average y of two points, ignoring None."""
    pts = [p for p in [p1, p2] if p is not None]
    return sum(p["y"] for p in pts) / len(pts) if pts else None


# ---------------------------------------------------------------------------
# Deadlift checks
# ---------------------------------------------------------------------------

def check_dl_rounded_back(keypoints):
    """
    Fault if the hip → shoulder → nose angle deviates > 15° from straight (180°).
    Check both sides and flag if either shows rounding.
    """
    faults = []
    for side in ("left", "right"):
        hip = kp(keypoints, f"{side}_hip")
        shoulder = kp(keypoints, f"{side}_shoulder")
        nose = kp(keypoints, "nose")
        angle = angle_at_joint(hip, shoulder, nose)
        if angle is not None and angle < 165:
            faults.append({
                "id": "DL_ROUNDED_BACK",
                "description": "Lower back is rounding — spine is not neutral.",
                "cue": "Chest up, proud chest. Pull the slack out of the bar before driving through the floor.",
            })
            break  # one fault per check regardless of side
    return faults


def check_dl_hips_rise_first(keypoints_sequence):
    """
    Fault if hips rise faster than shoulders across a sequence of frames.
    In image coords y increases downward, so rising = y decreasing.
    keypoints_sequence: list of keypoint dicts ordered earliest → latest.
    """
    if not keypoints_sequence or len(keypoints_sequence) < 2:
        return []

    first, last = keypoints_sequence[0], keypoints_sequence[-1]

    l_hip_f = kp(first, "left_hip")
    r_hip_f = kp(first, "right_hip")
    l_sho_f = kp(first, "left_shoulder")
    r_sho_f = kp(first, "right_shoulder")

    l_hip_l = kp(last, "left_hip")
    r_hip_l = kp(last, "right_hip")
    l_sho_l = kp(last, "left_shoulder")
    r_sho_l = kp(last, "right_shoulder")

    hip_y_start = avg_y(l_hip_f, r_hip_f)
    hip_y_end   = avg_y(l_hip_l, r_hip_l)
    sho_y_start = avg_y(l_sho_f, r_sho_f)
    sho_y_end   = avg_y(l_sho_l, r_sho_l)

    if None in (hip_y_start, hip_y_end, sho_y_start, sho_y_end):
        return []

    # Rise = decrease in y. Negative delta = rose.
    hip_rise = hip_y_start - hip_y_end
    sho_rise = sho_y_start - sho_y_end

    if hip_rise > sho_rise + 0.02:  # hips rose meaningfully more than shoulders
        return [{
            "id": "DL_HIPS_RISE_FIRST",
            "description": "Hips are shooting up before the chest — 'stripper deadlift'.",
            "cue": "Push the floor away. Imagine leg-pressing the ground while keeping your chest angle the same off the floor.",
        }]
    return []


def check_dl_bar_drift(keypoints):
    """
    Fault if the wrist is too far horizontally from the ankle (midfoot proxy).
    """
    faults = []
    for side in ("left", "right"):
        wrist = kp(keypoints, f"{side}_wrist")
        ankle = kp(keypoints, f"{side}_ankle")
        if wrist is not None and ankle is not None:
            drift = abs(wrist["x"] - ankle["x"])
            if drift > BAR_DRIFT_THRESHOLD:
                faults.append({
                    "id": "DL_BAR_DRIFT",
                    "description": "Bar is drifting away from the body.",
                    "cue": "Drag the bar up your shins. Think 'bar over mid-foot' throughout the pull.",
                })
                break
    return faults


def check_dl_squat_setup(keypoints):
    """
    Fault if the hip angle (knee → hip → shoulder) < 45°, meaning hips are
    set too low (squat-like rather than a hip hinge).
    """
    for side in ("left", "right"):
        knee     = kp(keypoints, f"{side}_knee")
        hip      = kp(keypoints, f"{side}_hip")
        shoulder = kp(keypoints, f"{side}_shoulder")
        angle    = angle_at_joint(knee, hip, shoulder)
        if angle is not None and angle < 45:
            return [{
                "id": "DL_SQUAT_SETUP",
                "description": "Hips are set too low at setup — treating it like a squat.",
                "cue": "Hinge, don't squat. Push your hips back, not straight down. Bar should be over mid-foot.",
            }]
    return []


# ---------------------------------------------------------------------------
# Back squat checks
# ---------------------------------------------------------------------------

def check_sq_knee_cave(keypoints):
    """
    Fault if either knee x-coordinate crosses inside the ankle x-coordinate.
    Left knee caving: left_knee.x > left_ankle.x (moves toward center/right).
    Right knee caving: right_knee.x < right_ankle.x (moves toward center/left).
    """
    faults = []

    l_knee = kp(keypoints, "left_knee")
    l_ankle = kp(keypoints, "left_ankle")
    if l_knee and l_ankle and l_knee["x"] > l_ankle["x"]:
        faults.append({
            "id": "SQ_KNEE_CAVE",
            "description": "Knees are caving inward (valgus collapse).",
            "cue": "Screw your feet into the floor and drive your knees out over your pinky toes.",
        })
        return faults  # one fault per check

    r_knee = kp(keypoints, "right_knee")
    r_ankle = kp(keypoints, "right_ankle")
    if r_knee and r_ankle and r_knee["x"] < r_ankle["x"]:
        faults.append({
            "id": "SQ_KNEE_CAVE",
            "description": "Knees are caving inward (valgus collapse).",
            "cue": "Screw your feet into the floor and drive your knees out over your pinky toes.",
        })

    return faults


def check_sq_depth(keypoints):
    """
    Fault if hip crease has not dropped below the knee at the bottom.
    In image coords y increases downward, so below = higher y value.
    """
    for side in ("left", "right"):
        hip   = kp(keypoints, f"{side}_hip")
        knee  = kp(keypoints, f"{side}_knee")
        if hip and knee and hip["y"] <= knee["y"]:
            return [{
                "id": "SQ_DEPTH",
                "description": "Squat is not hitting depth — hip crease is not below the knee.",
                "cue": "Keep descending until your hip crease breaks the plane of the top of your knee.",
            }]
    return []


def check_sq_forward_lean(keypoints):
    """
    Fault if the torso (hip → shoulder) leans more than 45° from vertical.
    """
    for side in ("left", "right"):
        hip      = kp(keypoints, f"{side}_hip")
        shoulder = kp(keypoints, f"{side}_shoulder")
        angle    = vertical_angle(hip, shoulder)
        if angle is not None and angle > 45:
            return [{
                "id": "SQ_FORWARD_LEAN",
                "description": "Excessive forward torso lean at the bottom of the squat.",
                "cue": "Chest up, elbows down. Brace your core and think 'tall spine' out of the hole.",
            }]
    return []


def check_sq_heel_rise(keypoints_sequence):
    """
    Fault if ankle y-coordinate shifts significantly across frames (heels rising).
    In image coords, heels rising = ankle y decreasing.
    """
    if not keypoints_sequence or len(keypoints_sequence) < 2:
        return []

    first, last = keypoints_sequence[0], keypoints_sequence[-1]

    for side in ("left", "right"):
        ankle_f = kp(first, f"{side}_ankle")
        ankle_l = kp(last,  f"{side}_ankle")
        if ankle_f and ankle_l:
            shift = abs(ankle_f["y"] - ankle_l["y"])
            if shift > HEEL_RISE_THRESHOLD:
                return [{
                    "id": "SQ_HEEL_RISE",
                    "description": "Heels are rising off the floor during the descent.",
                    "cue": "Drive through your whole foot. Consider ankle mobility work or elevate heels slightly.",
                }]
    return []


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def check_form(keypoints, lift_type, keypoints_sequence=None):
    """
    Run all form checks for the given lift.

    Args:
        keypoints: Dict of joint name → {"x", "y", "confidence"} for the key frame.
        lift_type: "deadlift" or "back_squat".
        keypoints_sequence: Optional list of keypoint dicts across frames, used
                            for checks that require temporal data (DL_HIPS_RISE_FIRST,
                            SQ_HEEL_RISE). Pass None to skip those checks.

    Returns:
        List of fault dicts: [{"id", "description", "cue"}, ...]
    """
    faults = []

    if lift_type == "deadlift":
        faults += check_dl_rounded_back(keypoints)
        faults += check_dl_bar_drift(keypoints)
        faults += check_dl_squat_setup(keypoints)
        if keypoints_sequence:
            faults += check_dl_hips_rise_first(keypoints_sequence)

    elif lift_type == "back_squat":
        faults += check_sq_knee_cave(keypoints)
        faults += check_sq_depth(keypoints)
        faults += check_sq_forward_lean(keypoints)
        if keypoints_sequence:
            faults += check_sq_heel_rise(keypoints_sequence)

    else:
        raise ValueError(f"Unknown lift type: '{lift_type}'. Use 'deadlift' or 'back_squat'.")

    return faults


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def pt(x, y, conf=0.9):
        return {"x": x, "y": y, "confidence": conf}

    # --- Deadlift test: rounded back + squat setup ---
    # Image coords: y increases downward, origin top-left, range 0–1.
    # Athlete viewed from the side. Hips low (squat setup), spine rounded.
    dl_keypoints = {
        "nose":           pt(0.50, 0.10),
        "left_shoulder":  pt(0.45, 0.30),
        "right_shoulder": pt(0.55, 0.30),
        "left_hip":       pt(0.48, 0.60),   # hips low → squat setup
        "right_hip":      pt(0.52, 0.60),
        "left_knee":      pt(0.45, 0.75),
        "right_knee":     pt(0.55, 0.75),
        "left_ankle":     pt(0.44, 0.90),
        "right_ankle":    pt(0.56, 0.90),
        "left_wrist":     pt(0.35, 0.65),   # wrist far from ankle → bar drift
        "right_wrist":    pt(0.65, 0.65),
    }
    # Sequence: hips rise faster than shoulders
    dl_frame2 = {k: dict(v) for k, v in dl_keypoints.items()}
    dl_frame2["left_hip"]["y"]       = 0.45   # hips shot up
    dl_frame2["right_hip"]["y"]      = 0.45
    dl_frame2["left_shoulder"]["y"]  = 0.28   # shoulders barely moved
    dl_frame2["right_shoulder"]["y"] = 0.28

    print("=== DEADLIFT TEST ===")
    dl_faults = check_form(dl_keypoints, "deadlift", keypoints_sequence=[dl_keypoints, dl_frame2])
    if dl_faults:
        for f in dl_faults:
            print(f"  [{f['id']}] {f['description']}")
            print(f"         Cue: {f['cue']}")
    else:
        print("  No faults detected.")

    print()

    # --- Back squat test: knee cave + no depth + forward lean ---
    sq_keypoints = {
        "nose":           pt(0.50, 0.10),
        "left_shoulder":  pt(0.40, 0.30),   # leaning far forward
        "right_shoulder": pt(0.60, 0.30),
        "left_hip":       pt(0.46, 0.55),   # hip not below knee → no depth
        "right_hip":      pt(0.54, 0.55),
        "left_knee":      pt(0.50, 0.70),   # left knee caved inside left ankle
        "right_knee":     pt(0.50, 0.70),   # right knee caved inside right ankle
        "left_ankle":     pt(0.42, 0.88),
        "right_ankle":    pt(0.58, 0.88),
    }
    # Sequence: heels rise during descent
    sq_frame2 = {k: dict(v) for k, v in sq_keypoints.items()}
    sq_frame2["left_ankle"]["y"]  = 0.84   # ankles rose
    sq_frame2["right_ankle"]["y"] = 0.84

    print("=== BACK SQUAT TEST ===")
    sq_faults = check_form(sq_keypoints, "back_squat", keypoints_sequence=[sq_keypoints, sq_frame2])
    if sq_faults:
        for f in sq_faults:
            print(f"  [{f['id']}] {f['description']}")
            print(f"         Cue: {f['cue']}")
    else:
        print("  No faults detected.")
