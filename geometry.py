import cv2
import numpy as np
import math


# make sure the 4 points are in the right order
def order_points(pts):
    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


# warp the card to a straight, fixed view
def warp_card(image, rect, out_w=200, out_h=300):
    box = cv2.boxPoints(rect)
    box = order_points(box)

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))

    return warped


# convert detected card into required output format
def get_card_geometry(rect, img_shape):
    (cx, cy), (rw, rh), _ = rect
    h, w = img_shape[:2]

    pts = cv2.boxPoints(rect)
    x, y, bw, bh = cv2.boundingRect(np.int32(pts))

    # normalised values (0 to 1)
    ccx = cx / w
    ccy = cy / h
    bbw = bw / w
    bbh = bh / h
    crw = rw / w
    crh = rh / h

    # work out rotation angle using longest side
    pts = order_points(pts)
    vec1 = pts[1] - pts[0]
    vec2 = pts[3] - pts[0]

    long_vec = vec1 if np.linalg.norm(vec1) > np.linalg.norm(vec2) else vec2

    # convert to "anticlockwise from vertical"
    angle_x = math.degrees(math.atan2(long_vec[1], long_vec[0]))
    angle_from_vertical = (90 - angle_x) % 360

    return {
        "ccx": ccx,
        "ccy": ccy,
        "bbw": bbw,
        "bbh": bbh,
        "crw": crw,
        "crh": crh,
        "angle": float(angle_from_vertical),
        "bbox": (x, y, bw, bh)
    }