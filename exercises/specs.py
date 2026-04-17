SPECS = {
  'squat': {
    'thresholds': {
      'top_knee_min': 160,
      'bottom_knee_max': 90,
      'hold_frames': 2
    },
    'form_rules': [
      {"id":"depth_ok","label":"Depth too shallow","metric":"knee_angle_min","op":"<=","value":90},
      {"id":"back_neutral","label":"Back angle out of range","metric":"trunk_angle","range":[10,45]}
    ]
  },
  'pushup': {
    'thresholds': {
      'top_elbow_min': 140,
      'bottom_elbow_max': 120,
      'hold_frames': 1,
      'min_trunk_angle': 20,
      'min_knee_angle': 110,
      'ready_frames': 1,
      'arm_frames': 1,
      'min_rep_gap_frames': 10,
      'min_cycle_frames': 8
    },
    'form_rules': [
      {"id":"rom_ok","label":"Range of motion too short","metric":"elbow_angle_min","op":"<=","value":75}
    ]
  },
  'bicep_curl': {
    'thresholds': {
      'top_elbow_max': 75,
      'bottom_elbow_min': 145,
      'hold_frames': 1,
      'max_trunk_angle': 25
    },
    'form_rules': [
        {"id":"curl_peak_ok","label":"Not curling high enough","metric":"elbow_angle_min","op":"<=","value":75,"states":["TOP"]},
        {"id":"arm_extend_ok","label":"Not extending arm enough","metric":"elbow_angle_min","op":">=","value":135,"states":["BOTTOM"]},
        {"id":"torso_stable","label":"Body swing too much","metric":"trunk_angle","range":[0,25]}
    ]
  },
  'pullup': {
    'thresholds': {
      'top_elbow_max': 85,
      'bottom_elbow_min': 135,
      'hold_frames': 1
    },
    'form_rules': []
  }
}
