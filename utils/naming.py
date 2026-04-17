# Landmark names and skeleton connections for MediaPipe Pose
MEDIAPIPE_LANDMARKS = [
    'nose',
    'left_eye_inner','left_eye','left_eye_outer','right_eye_inner','right_eye','right_eye_outer',
    'left_ear','right_ear',
    'mouth_left','mouth_right',
    'left_shoulder','right_shoulder',
    'left_elbow','right_elbow',
    'left_wrist','right_wrist',
    'left_pinky','right_pinky',
    'left_index','right_index',
    'left_thumb','right_thumb',
    'left_hip','right_hip',
    'left_knee','right_knee',
    'left_ankle','right_ankle',
    'left_heel','right_heel',
    'left_foot_index','right_foot_index'
]

# Simple connections for overlay (subset)
SKELETON_EDGES = [
    ('left_shoulder','right_shoulder'),
    ('left_hip','right_hip'),
    ('left_shoulder','left_elbow'),('left_elbow','left_wrist'),
    ('right_shoulder','right_elbow'),('right_elbow','right_wrist'),
    ('left_hip','left_knee'),('left_knee','left_ankle'),
    ('right_hip','right_knee'),('right_knee','right_ankle'),
    ('left_shoulder','left_hip'),('right_shoulder','right_hip')
]

NAME2IDX = {name:i for i,name in enumerate(MEDIAPIPE_LANDMARKS)}
