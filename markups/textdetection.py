import cv2
import numpy as np

def detect_text(src):
    """
    Perform the text detection algorithm and return a list of regions, where each region contains a word in the text.

    Args:
        src (np.array): source BGR image
    Returns:
        List[Tuple[int]]: A list of bounding boxes of the form (x, y, w, h)
              where each bounding box contains a word detected in the text.
              x is the center x coordinate, y is the center y coordinate,
              w is the width, and h is the height.
    """
    binary = _binarize_with_morph_grad(src)
    contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))

    regions = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # to draw the contours on the source image
        # cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cropped = binary[y:y+h, x:x+w]

        # ratio of non-zero pixels in filled region
        r = float(cv2.countNonZero(cropped)) / (cropped.shape[0] * cropped.shape[1])
        if r > .3 and cropped.shape[0] > 5 and cropped.shape[1] > 5: # constrain region size
            regions.append((x, y, w, h))

    return regions

def _binarize_with_morph_grad(src):
    """
    First perform morphological gradient, then use Otsu's binarization (https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#otsus-binarization)
    """
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # apply morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    gradient = cv2.morphologyEx(src_gray, cv2.MORPH_GRADIENT, kernel)
    gradient = gradient.astype(np.uint8)

    # make black and white
    thresh_used, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

