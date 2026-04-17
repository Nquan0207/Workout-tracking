import numpy as np

EPS = 1e-6

def angle_three_points(a, b, c):
    a, b, c = np.array(a[:2], float), np.array(b[:2], float), np.array(c[:2], float)
    ba, bc = a-b, c-b
    num = float(ba @ bc)
    den = float(np.linalg.norm(ba) * np.linalg.norm(bc) + EPS)
    cosang = np.clip(num/den, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def vector_angle_degrees(u, v):
    u, v = np.array(u[:2], float), np.array(v[:2], float)
    num = float(u @ v)
    den = float(np.linalg.norm(u) * np.linalg.norm(v) + EPS)
    cosang = np.clip(num/den, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def trunk_forward_lean(shoulder, hip):
    # angle between trunk vector (hip->shoulder) and vertical axis (0,-1)
    vec = np.array([shoulder[0]-hip[0], shoulder[1]-hip[1]], float)
    vertical = np.array([0.0, -1.0], float)
    return vector_angle_degrees(vec, vertical)
