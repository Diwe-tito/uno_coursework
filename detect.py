import cv2
import numpy as np


def offset_contour(cnt, ox, oy):
    cnt = cnt.copy()
    cnt[:, 0, 0] += ox
    cnt[:, 0, 1] += oy
    return cnt


def touches_border(x, y, w, h, img_w, img_h, margin=6):
    return (
        x <= margin or
        y <= margin or
        x + w >= img_w - margin or
        y + h >= img_h - margin
    )


def contour_overlap_ratio(cnt, mask):
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    area = cv2.countNonZero(contour_mask)
    if area == 0:
        return 0.0
    overlap = cv2.countNonZero(cv2.bitwise_and(contour_mask, mask))
    return overlap / float(area)


def shrink_contour(cnt, scale=0.82):
    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-6:
        return cnt.copy()

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    pts = cnt[:, 0, :].astype(np.float32)
    pts[:, 0] = cx + (pts[:, 0] - cx) * scale
    pts[:, 1] = cy + (pts[:, 1] - cy) * scale

    return pts.reshape((-1, 1, 2)).astype(np.int32)


# try to find UNO white ovals inside a merged blob
def _find_ovals_in_blob(cnt, white_mask, nominal_card_area):
    x, y, w, h = cv2.boundingRect(cnt)

    pad = 4
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(white_mask.shape[1], x + w + pad)
    y1 = min(white_mask.shape[0], y + h + pad)

    blob_mask = np.zeros((y1 - y0, x1 - x0), np.uint8)

    shifted = cnt.copy()
    shifted[:, 0, 0] -= x0
    shifted[:, 0, 1] -= y0
    cv2.drawContours(blob_mask, [shifted], -1, 255, cv2.FILLED)

    white_in = cv2.bitwise_and(white_mask[y0:y1, x0:x1], blob_mask)

    card_w = int(min(white_mask.shape[:2]) * 0.13)

    # first option: erode then grow back
    bt = max(3, card_w // 10)
    ek = np.ones((bt * 2 + 1, bt * 2 + 1), np.uint8)
    proc_e = cv2.dilate(cv2.erode(white_in, ek, iterations=1), ek, iterations=1)
    ovals_e = _score_oval_contours(proc_e, cv2.RETR_EXTERNAL, nominal_card_area, x0, y0)

    # second option: close gaps in the oval
    ck = max(5, card_w // 8) | 1
    k = np.ones((ck, ck), np.uint8)
    proc_c = cv2.morphologyEx(white_in, cv2.MORPH_CLOSE, k, iterations=3)
    ovals_c = _score_oval_contours(proc_c, cv2.RETR_LIST, nominal_card_area, x0, y0)

    # keep whichever gives more results
    return ovals_e if len(ovals_e) >= len(ovals_c) else ovals_c


def _score_oval_contours(proc, retr_mode, nominal_card_area, x0, y0):
    sub_cnts, _ = cv2.findContours(proc, retr_mode, cv2.CHAIN_APPROX_SIMPLE)

    ovals = []
    for sc in sub_cnts:
        area = cv2.contourArea(sc)

        # expected oval area relative to a card
        if area < nominal_card_area * 0.12 or area > nominal_card_area * 0.50:
            continue

        perim = cv2.arcLength(sc, True)
        if perim < 1:
            continue

        circularity = 4 * np.pi * area / (perim * perim)
        if circularity < 0.50:
            continue

        rect = cv2.minAreaRect(sc)
        (_, _), (rw, rh), angle = rect

        if rw <= 0 or rh <= 0:
            continue

        ratio = max(rw, rh) / min(rw, rh)
        if ratio < 1.0 or ratio > 1.95:
            continue

        # expand oval to roughly card size
        card_rect = (
            (rect[0][0] + x0, rect[0][1] + y0),
            (rw * 2.0, rh * 2.0),
            angle
        )
        ovals.append(card_rect)

    return ovals


# split blobs that may contain more than one card
def split_merged_contour(cnt, mask, white_mask, nominal_card_area):
    blob_area = cv2.contourArea(cnt)

    # estimate likely card area from the current image
    all_areas = sorted(
        cv2.contourArea(c)
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if cv2.contourArea(c) > nominal_card_area * 0.5
    )
    est_card_area = all_areas[0] if all_areas else nominal_card_area

    # only try oval split if blob looks big enough
    if blob_area > est_card_area * 2.0:
        oval_rects = _find_ovals_in_blob(cnt, white_mask, nominal_card_area)
        if len(oval_rects) >= 2:
            return [{"rect": r, "from_oval": True} for r in oval_rects]

    # fallback: erode and separate connected parts
    x, y, w, h = cv2.boundingRect(cnt)

    pad = 4
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(mask.shape[1], x + w + pad)
    y1 = min(mask.shape[0], y + h + pad)

    roi_mask = mask[y0:y1, x0:x1]

    local = np.zeros_like(roi_mask)
    shifted = cnt.copy()
    shifted[:, 0, 0] -= x0
    shifted[:, 0, 1] -= y0
    cv2.drawContours(local, [shifted], -1, 255, thickness=cv2.FILLED)

    erode_kernel = np.ones((5, 5), np.uint8)
    original_area = cv2.contourArea(cnt)

    for it in (12, 10, 8, 6, 4):
        eroded = cv2.erode(local, erode_kernel, iterations=it)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            eroded, connectivity=8
        )

        parts = []
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < max(80, 0.06 * original_area):
                continue

            comp = np.zeros_like(eroded)
            comp[labels == i] = 255

            grow_iters = 3 if it >= 8 else 2
            grown = cv2.dilate(comp, erode_kernel, iterations=grow_iters)
            grown = cv2.bitwise_and(grown, local)

            sub_contours, _ = cv2.findContours(
                grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not sub_contours:
                continue

            best = max(sub_contours, key=cv2.contourArea)
            if cv2.contourArea(best) < 0.10 * original_area:
                continue

            parts.append(offset_contour(best, x0, y0))

        if len(parts) >= 2:
            return parts

    return [cnt]


# build masks used for detection
def make_masks(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    l = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    hsv_eq = cv2.merge([h, s, v_eq])

    # white areas
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 65, 255])
    white_hsv = cv2.inRange(hsv, lower_white, upper_white)

    white_adaptive = np.zeros_like(v, dtype=np.uint8)
    white_adaptive[(v_eq > 140) & (s < 90)] = 255

    # coloured card regions
    red1 = cv2.inRange(hsv_eq, np.array([0, 60, 40]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv_eq, np.array([168, 60, 40]), np.array([180, 255, 255]))
    yellow = cv2.inRange(hsv_eq, np.array([18, 50, 50]), np.array([42, 255, 255]))
    green = cv2.inRange(hsv_eq, np.array([40, 25, 25]), np.array([95, 255, 255]))
    blue = cv2.inRange(hsv_eq, np.array([90, 25, 25]), np.array([140, 255, 255]))

    colour_mask = red1 | red2 | yellow | green | blue
    colour_mask = cv2.morphologyEx(
        colour_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )

    # darker parts used for wildcards
    dark_mask = np.zeros_like(v, dtype=np.uint8)
    dark_mask[(v_eq < 110) & (s > 25)] = 255

    white_support = cv2.dilate(white_hsv, np.ones((9, 9), np.uint8), iterations=1)
    dark_supported = cv2.bitwise_and(dark_mask, white_support)

    face_support = cv2.bitwise_or(colour_mask, dark_supported)
    face_support = cv2.dilate(face_support, np.ones((11, 11), np.uint8), iterations=1)

    white_lab = np.zeros_like(l, dtype=np.uint8)
    white_lab[(l > 150) & (s < 85)] = 255
    white_lab = cv2.bitwise_and(white_lab, face_support)

    white_mask = cv2.bitwise_or(white_hsv, white_lab)
    white_mask = cv2.bitwise_or(white_mask, white_adaptive)

    mask = cv2.bitwise_or(white_mask, dark_supported)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    return white_mask, mask, closing, face_support


# decide whether a contour looks enough like a card
def valid_card_contour(cnt, image_shape, face_support, white_mask):
    img_h, img_w = image_shape[:2]
    img_area = img_h * img_w

    area = cv2.contourArea(cnt)
    if area < (img_area * 0.0035):
        return False
    if area > (img_area * 0.10):
        return False

    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    if rect_area <= 0:
        return False

    border_touch = touches_border(x, y, w, h, img_w, img_h)

    extent = float(area) / rect_area
    min_extent = 0.32 if not border_touch else 0.16
    if extent < min_extent:
        return False

    rect = cv2.minAreaRect(cnt)
    (_, _), (rw, rh), _ = rect
    if rw <= 0 or rh <= 0:
        return False

    ratio = max(rw, rh) / min(rw, rh)
    min_ratio = 1.05 if not border_touch else 0.45
    max_ratio = 2.20 if not border_touch else 4.20
    if ratio < min_ratio or ratio > max_ratio:
        return False

    fill_ratio = area / float((rw * rh) + 1e-6)
    min_fill = 0.42 if not border_touch else 0.18
    if fill_ratio < min_fill:
        return False

    support_ratio = contour_overlap_ratio(cnt, face_support)
    min_support = 0.09 if not border_touch else 0.025
    if support_ratio < min_support:
        return False

    inner = shrink_contour(cnt, 0.82)
    inner_support_ratio = contour_overlap_ratio(inner, face_support)
    min_inner_support = 0.07 if not border_touch else 0.015
    if inner_support_ratio < min_inner_support:
        return False

    return True


# basic IoU for duplicate removal
def iou_rect(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union


# main detection function
def detect_cards(image):
    white_mask, mask, closing, face_support = make_masks(image)

    contours, _ = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    nominal_card_area = image.shape[0] * image.shape[1] * 0.028
    candidates = []

    for cnt in contours:
        split_parts = split_merged_contour(
            cnt, closing, white_mask, nominal_card_area
        )

        for part in split_parts:
            # oval split returns rects directly
            if isinstance(part, dict):
                rect = part["rect"]
                box = cv2.boxPoints(rect).astype(np.int32)
                x, y, w, h = cv2.boundingRect(box)

                candidates.append({
                    "rect": rect,
                    "bbox": (x, y, w, h),
                    "area": nominal_card_area
                })
                continue

            if not valid_card_contour(part, image.shape, face_support, white_mask):
                continue

            rect = cv2.minAreaRect(part)
            box = cv2.boxPoints(rect).astype(np.int32)
            x, y, w, h = cv2.boundingRect(box)

            candidates.append({
                "rect": rect,
                "bbox": (x, y, w, h),
                "area": cv2.contourArea(part)
            })

    # keep larger detections first
    candidates.sort(key=lambda c: c["area"], reverse=True)

    cards = []
    kept = []

    for c in candidates:
        duplicate = False
        for k in kept:
            if iou_rect(c["bbox"], k) > 0.35:
                duplicate = True
                break

        if not duplicate:
            cards.append({"rect": c["rect"]})
            kept.append(c["bbox"])

    debug_data = {
        "gray": white_mask,
        "edges": closing,
        "colour_mask": face_support
    }

    return cards, debug_data