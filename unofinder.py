import sys
import os
import cv2

from detect import detect_cards
from geometry import get_card_geometry
from utils import save_debug, reset_output
from identify import identify_card


TEMPLATE_DIR = os.path.join("images", "templates")


# load input image
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image: {path}")
        sys.exit(1)
    return img


# keep template names in coursework format
def normalise_template_label(label):
    label = label.lower().strip()
    label = label.replace("plus2", "plustwo")
    label = label.replace("plus_two", "plustwo")
    label = label.replace("plus4", "plusfour")
    label = label.replace("plus_four", "plusfour")
    return label


# load all template images
def load_templates(template_dir=TEMPLATE_DIR):
    templates = {}

    if not os.path.isdir(template_dir):
        print(f"Template folder not found: {template_dir}")
        return templates

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")

    for fname in os.listdir(template_dir):
        if not fname.lower().endswith(valid_exts):
            continue

        label = normalise_template_label(os.path.splitext(fname)[0])
        path = os.path.join(template_dir, fname)

        img = cv2.imread(path)
        if img is None:
            continue

        templates[label] = img

    return templates


# write final detections to output.txt
def write_output(filename, detections):
    reset_output()

    with open("output.txt", "w") as f:
        for d in detections:
            line = (
                f"{filename},{d.get('label', 'unknown')},"
                f"{d['ccx']:.6f},{d['ccy']:.6f},"
                f"{d['bbw']:.6f},{d['bbh']:.6f},"
                f"{d['crw']:.6f},{d['crh']:.6f},"
                f"{d['angle']:.2f}\n"
            )
            f.write(line)


def main():
    if len(sys.argv) < 2:
        print('Usage: python unofinder.py "path/to/inputfile.jpg" [debug]')
        sys.exit(1)

    image_path = sys.argv[1]
    debug = len(sys.argv) > 2 and sys.argv[2].lower() == "debug"

    filename = os.path.basename(image_path)
    img = load_image(image_path)

    templates = load_templates()
    cards, debug_data = detect_cards(img)

    detections = []
    debug_img = img.copy()

    for card in cards:
        rect = card["rect"]

        info = identify_card(img, rect, templates=templates if templates else None)
        geom = get_card_geometry(rect, img.shape)

        geom["label"] = info["label"]
        detections.append(geom)

        if debug:
            x, y, w, h = geom["bbox"]
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                debug_img,
                geom["label"],
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    write_output(filename, detections)

    save_debug(debug, "gray", debug_data["gray"])
    save_debug(debug, "edges", debug_data["edges"])
    save_debug(debug, "mask", debug_data["colour_mask"])
    save_debug(debug, "detections", debug_img)

    if debug:
        total_cards = len(detections)
        unknown_cards = sum(1 for d in detections if d["label"] == "unknown")

        print(f"Image: {filename}")
        print(f"Detected card candidates: {total_cards}")
        print(f"Unknown labels: {unknown_cards}")
        print("Created: output.txt, debug_gray.png, debug_edges.png, debug_colour_mask.png, debug_detections.png")


if __name__ == "__main__":
    main()