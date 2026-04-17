from .geometry import angle_three_points, trunk_forward_lean

def knee_angle(side, kps):
    hip = kps[f'{side}_hip']
    knee = kps[f'{side}_knee']
    ankle = kps[f'{side}_ankle']
    return angle_three_points(hip, knee, ankle)

def elbow_angle(side, kps):
    shoulder = kps[f'{side}_shoulder']
    elbow = kps[f'{side}_elbow']
    wrist = kps[f'{side}_wrist']
    return angle_three_points(shoulder, elbow, wrist)

def trunk_angle(kps):
    # mean of left/right forward lean
    left = trunk_forward_lean(kps['left_shoulder'], kps['left_hip'])
    right = trunk_forward_lean(kps['right_shoulder'], kps['right_hip'])
    return (left + right) / 2.0
