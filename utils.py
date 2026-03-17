import cv2
import os


# save debug images when debug mode is enabled
def save_debug(debug, name, img):
    if debug:
        cv2.imwrite(f"debug_{name}.png", img)


# remove old output file before writing a new one
def reset_output():
    if os.path.exists("output.txt"):
        os.remove("output.txt")