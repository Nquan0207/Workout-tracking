import cv2
from ..utils.naming import SKELETON_EDGES

def draw_pose(frame, kps):
    # kps: name -> (x,y,z,v)
    for a,b in SKELETON_EDGES:
        if a in kps and b in kps:
            xa,ya = int(kps[a][0]), int(kps[a][1])
            xb,yb = int(kps[b][0]), int(kps[b][1])
            cv2.line(frame, (xa,ya), (xb,yb), (0,255,0), 2)
    for name,(x,y,_,v) in kps.items():
        if v > 0.5:
            cv2.circle(frame, (int(x),int(y)), 3, (255,255,255), -1)
    return frame

def draw_hud(frame, text_lines, org=(10,30)):
    x,y = org
    for i, t in enumerate(text_lines):
        baseline = y + i*22
        cv2.putText(frame, t, (x, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(frame, t, (x, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    return frame
