import cv2
import numpy as np

# standard card size after warping
CARD_W = 200
CARD_H = 300

# feature detector + matcher
AKAZE = cv2.AKAZE_create()
BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


# order points so perspective transform works correctly
def order_box_points(pts):
    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    return np.array([
        pts[np.argmin(s)],      # top-left
        pts[np.argmin(diff)],   # top-right
        pts[np.argmax(s)],      # bottom-right
        pts[np.argmax(diff)]    # bottom-left
    ], dtype=np.float32)


# warp detected card into a clean upright view
def warp_card(image, rect):
    box = cv2.boxPoints(rect)
    box = order_box_points(box)

    dst = np.array([
        [0, 0],
        [CARD_W - 1, 0],
        [CARD_W - 1, CARD_H - 1],
        [0, CARD_H - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(image, M, (CARD_W, CARD_H))

    # fix orientation if sideways
    if warped.shape[1] > warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped


# classify card colour using HSV
def classify_colour(card_img):
    h, w = card_img.shape[:2]

    hsv = cv2.cvtColor(cv2.medianBlur(card_img, 5), cv2.COLOR_BGR2HSV)

    # check corners for dark regions (used for wildcards)
    patch_w = int(0.18 * w)
    patch_h = int(0.18 * h)

    corners = [
        hsv[0:patch_h, 0:patch_w],
        hsv[0:patch_h, w - patch_w:w],
        hsv[h - patch_h:h, 0:patch_w],
        hsv[h - patch_h:h, w - patch_w:w],
    ]

    dark_ratios = [
        cv2.countNonZero(cv2.inRange(p, (0, 0, 0), (180, 140, 110))) /
        float(p.shape[0] * p.shape[1] + 1e-6)
        for p in corners
    ]

    # focus on centre (ignore white oval)
    centre = hsv[int(0.22*h):int(0.78*h), int(0.22*w):int(0.78*w)]
    H, S, V = centre[:, :, 0], centre[:, :, 1], centre[:, :, 2]

    valid = (S > 55) & (V > 40)

    masks = {
        "red": (((H < 12) | (H > 168)) & valid),
        "yellow": ((H > 16) & (H < 45) & valid),
        "green": ((H > 42) & (H < 95) & valid),
        "blue": ((H > 90) & (H < 140) & valid),
    }

    # check how many colours appear (used for wildcards)
    hits = sum(np.count_nonzero(m) > 0.025 * centre.size for m in masks.values())

    if np.mean(dark_ratios) > 0.16 and hits >= 3:
        return "wildcard"

    # pick dominant colour
    scores = {k: np.count_nonzero(v) for k, v in masks.items()}
    return max(scores, key=scores.get)


# generate rotated versions (cards can be in any orientation)
def rotated_versions(img):
    return [
        cv2.resize(img, (CARD_W, CARD_H)),
        cv2.resize(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), (CARD_W, CARD_H)),
        cv2.resize(cv2.rotate(img, cv2.ROTATE_180), (CARD_W, CARD_H)),
        cv2.resize(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), (CARD_W, CARD_H))
    ]


# simple preprocessing before feature extraction
def preprocess_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.GaussianBlur(gray, (3, 3), 0)


# split card into useful regions
def extract_normal_rois(card):
    h, w = card.shape[:2]
    return (
        card[int(0.2*h):int(0.8*h), int(0.2*w):int(0.8*w)],   # centre
        card[0:int(0.28*h), 0:int(0.28*w)],                  # top-left
        card[int(0.72*h):h, int(0.72*w):w]                   # bottom-right
    )


def extract_wild_roi(card):
    h, w = card.shape[:2]
    return card[int(0.12*h):int(0.88*h), int(0.12*w):int(0.88*w)]


def roi_ready(img, size):
    return cv2.resize(preprocess_gray(img), size)


# compute AKAZE features
def compute_akaze(gray):
    return AKAZE.detectAndCompute(gray, None)


# compare two feature sets
def akaze_similarity(kpa, desa, kpb, desb):
    if desa is None or desb is None or len(kpa) < 2 or len(kpb) < 2:
        return 0.0

    matches = BF.knnMatch(desa, desb, k=2)

    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.78 * n.distance:
            good.append(m)

    if not good:
        return 0.0

    count_score = min(len(good) / 20.0, 1.0)
    mean_dist = np.mean([m.distance for m in good])
    dist_score = 1.0 - min(mean_dist / 256.0, 1.0)

    return 0.65 * count_score + 0.35 * dist_score


# quick template similarity (used to shortlist)
def corr_similarity(a, b):
    return float(cv2.matchTemplate(
        a.astype(np.float32),
        b.astype(np.float32),
        cv2.TM_CCOEFF_NORMED
    )[0, 0])


# filter templates by colour
def candidate_labels(labels, colour):
    if colour == "wildcard":
        return [l for l in labels if l in ["wildcard", "plusfour"]]
    return [l for l in labels if l.startswith(colour + "_")]


# build features for matching
def build_features(card, colour):
    if colour == "wildcard":
        center = roi_ready(extract_wild_roi(card), (140, 180))
        kp, des = compute_akaze(center)
        return {"center": center, "kp": kp, "des": des}

    c, tl, br = extract_normal_rois(card)

    c = roi_ready(c, (140, 180))
    tl = roi_ready(tl, (90, 110))
    br = roi_ready(br, (90, 110))

    return {
        "center": c, "tl": tl, "br": br,
        "kp_c": compute_akaze(c),
        "kp_tl": compute_akaze(tl),
        "kp_br": compute_akaze(br)
    }


# match card against templates
def match_card_to_templates(card, templates, colour):
    best_label = "unknown"
    best_score = -1.0

    labels = candidate_labels(list(templates.keys()), colour)
    if not labels:
        labels = list(templates.keys())

    for rot in rotated_versions(card):
        feat = build_features(rot, colour)

        for label in labels:
            tmpl = cv2.resize(templates[label], (CARD_W, CARD_H))
            feat_t = build_features(tmpl, colour)

            if colour == "wildcard":
                score = akaze_similarity(feat["kp"], feat["des"], feat_t["kp"], feat_t["des"])
            else:
                score = (
                    0.45 * akaze_similarity(*feat["kp_c"], *feat_t["kp_c"]) +
                    0.275 * akaze_similarity(*feat["kp_tl"], *feat_t["kp_tl"]) +
                    0.275 * akaze_similarity(*feat["kp_br"], *feat_t["kp_br"])
                )

            if score > best_score:
                best_score = score
                best_label = label

    # reject weak matches
    if best_score < 0.22:
        return "unknown", best_score
    if best_score < 0.28:
        return "unknown", best_score

    return best_label, best_score


# main function used by unofinder
def identify_card(image, rect, templates=None):
    warped = warp_card(image, rect)
    colour = classify_colour(warped)

    if templates:
        label, score = match_card_to_templates(warped, templates, colour)
    else:
        label, score = "unknown", 0.0

    # prevent mismatched colour labels
    if label != "unknown" and colour != "wildcard":
        if not label.startswith(colour):
            label = "unknown"

    # force wildcard consistency
    if colour == "wildcard" and label not in ["wildcard", "plusfour"]:
        label = "wildcard"

    return {
        "colour": colour,
        "label": label,
        "score": score,
        "warped": warped
    }