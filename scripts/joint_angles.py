import numpy as np

# COCO Keypoint Indices Mapping
KP = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle formed by three 2D points (x, y).
    p2 is the vertex of the angle.
    Returns angle in degrees (0-180).
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    # Cosine of angle is dot product divided by magnitudes
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    # Clip to avoid float inaccuracies causing exact bounding box out of domain for arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))

    return angle

def extract_key_angles(kps, conf_threshold=0.3):
    """
    Given (17, 3) keypoints [x, y, conf], compute various fitness-relevant angles.
    If any keypoint required for an angle is below conf_threshold,
    the angle is returned as None (to be handled/interpolated later).
    """
    angles = {}

    def get_pt(name):
        idx = KP[name]
        x, y, c = kps[idx]
        return np.array([x, y]), c

    # Left Knee Angle (Hip - Knee - Ankle)
    h_l, ch_l = get_pt('left_hip')
    k_l, ck_l = get_pt('left_knee')
    a_l, ca_l = get_pt('left_ankle')
    if ch_l > conf_threshold and ck_l > conf_threshold and ca_l > conf_threshold:
        angles['left_knee'] = calculate_angle(h_l, k_l, a_l)
    else:
        angles['left_knee'] = None

    # Right Knee Angle
    h_r, ch_r = get_pt('right_hip')
    k_r, ck_r = get_pt('right_knee')
    a_r, ca_r = get_pt('right_ankle')
    if ch_r > conf_threshold and ck_r > conf_threshold and ca_r > conf_threshold:
        angles['right_knee'] = calculate_angle(h_r, k_r, a_r)
    else:
        angles['right_knee'] = None

    # Left Shoulder-Hip-Knee (Torso flex)
    s_l, cs_l = get_pt('left_shoulder')
    if ch_l > conf_threshold and ck_l > conf_threshold and cs_l > conf_threshold:
        angles['left_hip'] = calculate_angle(s_l, h_l, k_l)
    else:
        angles['left_hip'] = None

    # Right Shoulder-Hip-Knee
    s_r, cs_r = get_pt('right_shoulder')
    if ch_r > conf_threshold and ck_r > conf_threshold and cs_r > conf_threshold:
        angles['right_hip'] = calculate_angle(s_r, h_r, k_r)
    else:
        angles['right_hip'] = None

    # Left Elbow (Shoulder - Elbow - Wrist)
    e_l, ce_l = get_pt('left_elbow')
    w_l, cw_l = get_pt('left_wrist')
    if cs_l > conf_threshold and ce_l > conf_threshold and cw_l > conf_threshold:
        angles['left_elbow'] = calculate_angle(s_l, e_l, w_l)
    else:
        angles['left_elbow'] = None

    return angles

def get_body_straightness(kps, conf_threshold=0.3):
    """
    Calculates body straightness for push-ups or planks.
    Averaged deviation from a straight 180-degree line between Shoulder -> Hip -> Ankle.
    Returns (left_deviation, right_deviation) where 0 is perfectly straight.
    """
    dev_l, dev_r = None, None
    s_l, c1 = get_pt_safe(kps, 'left_shoulder', conf_threshold)
    h_l, c2 = get_pt_safe(kps, 'left_hip', conf_threshold)
    a_l, c3 = get_pt_safe(kps, 'left_ankle', conf_threshold)
    if c1 and c2 and c3:
        ang = calculate_angle(s_l, h_l, a_l)
        dev_l = abs(180.0 - ang)

    s_r, c1 = get_pt_safe(kps, 'right_shoulder', conf_threshold)
    h_r, c2 = get_pt_safe(kps, 'right_hip', conf_threshold)
    a_r, c3 = get_pt_safe(kps, 'right_ankle', conf_threshold)
    if c1 and c2 and c3:
        ang = calculate_angle(s_r, h_r, a_r)
        dev_r = abs(180.0 - ang)

    return dev_l, dev_r

def get_pt_safe(kps, name, th):
    idx = KP[name]
    x, y, c = kps[idx]
    if c > th:
        return np.array([x, y]), True
    return np.array([0, 0]), False
