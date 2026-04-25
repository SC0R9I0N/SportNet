import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from model_backbone import get_keypoints, filename_to_image_id

# ------------------------------------------
# Utility functions
# ------------------------------------------

def angle(a, b, c):
    """Compute angle ABC in degrees."""
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def knee_angle(kpts, side):
    hip = kpts[11] if side == "left" else kpts[12]
    knee = kpts[13] if side == "left" else kpts[14]
    ankle = kpts[15] if side == "left" else kpts[16]
    return angle(hip[:2], knee[:2], ankle[:2])

def hip_height(kpts):
    left_hip = kpts[11][1]
    right_hip = kpts[12][1]
    return (left_hip + right_hip) / 2.0

def ankle_height(kpts):
    left_ankle = kpts[15][1]
    right_ankle = kpts[16][1]
    return (left_ankle + right_ankle) / 2.0

def stride_length(kpts):
    left_ankle = kpts[15][0]
    right_ankle = kpts[16][0]
    return abs(left_ankle - right_ankle)

def torso_angle(kpts):
    left_shoulder = kpts[5][:2]
    right_shoulder = kpts[6][:2]
    left_hip = kpts[11][:2]
    right_hip = kpts[12][:2]

    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    mid_hip = (left_hip + right_hip) / 2.0

    v = mid_shoulder - mid_hip
    vertical = np.array([0.0, -1.0])
    cosang = np.dot(v, vertical) / (np.linalg.norm(v) * np.linalg.norm(vertical) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def limb_symmetry(kpts, left_idx, right_idx, img_width=1280):
    left = kpts[left_idx][:2]
    right = kpts[right_idx][:2]
    return float(abs(left[0] - right[0]) / img_width)

# ------------------------------------------
# Rule-based classifier
# ------------------------------------------

def classify_pose(kpts, prev_kpts=None, bbox_height=None, bbox_width=None):
    """Return one of: standing, walking, running, squatting, jumping
    
    Args:
        kpts: keypoints array (17, 3)
        prev_kpts: previous frame keypoints for velocity
        bbox_height: bounding box height for normalization
        bbox_width: bounding box width for normalization
    """

    # Basic features
    hip_y = hip_height(kpts)
    ankle_y = ankle_height(kpts)
    stride = stride_length(kpts)
    left_knee = knee_angle(kpts, "left")
    right_knee = knee_angle(kpts, "right")

    # Normalize by bounding box dimensions (person-centric, not image-centric)
    hip_norm = hip_y / bbox_height if bbox_height else 0.5
    ankle_norm = ankle_y / bbox_height if bbox_height else 0.5
    stride_norm = stride / bbox_width if bbox_width else 0.5

    # Vertical velocity (if previous frame exists)
    vel = 0
    if prev_kpts is not None:
        vel = (hip_height(prev_kpts) - hip_y) / bbox_height if bbox_height else 0

    # ------------------------------------------
    # Rules
    # ------------------------------------------

    # Jumping: both feet off ground OR strong upward velocity
    if ankle_norm < 0.05 or vel > 0.02:
        return "jumping"

    # Squatting: hips low + BOTH knees bent
    if hip_norm > 0.55 and left_knee < 120 and right_knee < 120:
        return "squatting"

    # Running: large stride
    if stride_norm > 0.10:
        return "running"

    # Walking: moderate stride
    if 0.03 < stride_norm <= 0.10:
        return "walking"

    # Standing: default
    return "standing"

# ------------------------------------------
# Symbolic explanation generation
# ------------------------------------------

def build_symbolic_explanation(action, feats):
    lk, rk, hip_norm, ankle_norm, stride_norm, torso, arm_sym, leg_sym = feats

    parts = [f"The person is {action} because"]

    # Action-specific reasoning
    if action == "standing":
        parts.append(
            f" the hips are high (normalized height {hip_norm:.2f}), "
            f"the knees are nearly straight ({lk:.1f}° and {rk:.1f}°), "
            f"and the stride length is small ({stride_norm:.2f})."
        )
    elif action == "squatting":
        parts.append(
            f" the hips are low (normalized height {hip_norm:.2f}), "
            f"and both knees show deep flexion ({lk:.1f}° and {rk:.1f}°)."
        )
    elif action == "running":
        parts.append(
            f" the stride length is large ({stride_norm:.2f}), "
            f"the legs are relatively extended ({lk:.1f}° and {rk:.1f}°), "
            f"and the posture suggests forward motion."
        )
    elif action == "walking":
        parts.append(
            f" the stride length is moderate ({stride_norm:.2f}), "
            f"and the knees show moderate flexion ({lk:.1f}° and {rk:.1f}°)."
        )
    elif action == "jumping":
        parts.append(
            f" the ankles are elevated (normalized height {ankle_norm:.2f}), "
            f"and the posture suggests upward motion."
        )

    # Generic biomechanical commentary
    if torso > 20:
        parts.append(f" The torso is tilted by about {torso:.1f}° from vertical.")
    if arm_sym > 0.05:
        parts.append(f" The arms are spread horizontally (symmetry {arm_sym:.2f}).")
    if leg_sym > 0.05:
        parts.append(f" The legs are asymmetrically positioned (symmetry {leg_sym:.2f}).")

    return " ".join(parts)

# ------------------------------------------
# GT generation loop
# ------------------------------------------

def generate_gt(image_dir, output_json):
    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

    results = []
    prev_kpts = None
    prev_seq = None

    for fname in tqdm(img_files):
        path = os.path.join(image_dir, fname)
        kpts = get_keypoints(path)

        # Get actual image dimensions
        img = Image.open(path)
        img_width, img_height = img.size

        # Check sequence boundary and reset prev_kpts if needed
        seq_id = fname[0]
        if seq_id != prev_seq:
            prev_kpts = None
            prev_seq = seq_id

        # Use image dimensions as proxy for body-centric scale
        # (In ideal scenario, would use actual bounding box dimensions for person-relative normalization)
        label = classify_pose(kpts, prev_kpts, img_height, img_width)
        prev_kpts = kpts

        # Compute biomechanical features
        lk = knee_angle(kpts, "left")
        rk = knee_angle(kpts, "right")
        hip_y = hip_height(kpts)
        ankle_y = ankle_height(kpts)
        stride = stride_length(kpts)
        torso = torso_angle(kpts)
        arm_sym = limb_symmetry(kpts, 9, 10, img_width)
        leg_sym = limb_symmetry(kpts, 15, 16, img_width)

        # Normalize features by image dimensions (person height ≈ 0.4-0.5 of image height)
        hip_norm = float(hip_y / img_height)
        ankle_norm = float(ankle_y / img_height)
        stride_norm = float(stride / img_width)

        # Generate symbolic explanation
        feats = [lk, rk, hip_norm, ankle_norm, stride_norm, torso, arm_sym, leg_sym]
        explanation = build_symbolic_explanation(label, feats)

        results.append({
            "image_id": filename_to_image_id(fname),
            "file_name": fname,
            "keypoints": kpts.tolist(),
            "simple_action": label,
            "image_dims": [img_width, img_height],
            "left_knee_angle": float(lk),
            "right_knee_angle": float(rk),
            "hip_height_norm": hip_norm,
            "ankle_height_norm": ankle_norm,
            "stride_length_norm": stride_norm,
            "torso_angle": float(torso),
            "arm_symmetry": float(arm_sym),
            "leg_symmetry": float(leg_sym),
            "symbolic_explanation": explanation,
        })

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Saved GT labels, features, and explanations to {output_json}")


if __name__ == "__main__":
    # generate_gt("pose_2d/valid_set", "ap2d_simple_gt.json")
    generate_gt("pose_2d/train_set", "train_ap2d_simple_gt.json")
