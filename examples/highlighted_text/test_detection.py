import cv2
import numpy as np
from markups import (
        HighlightExtractor,
        detect_text,
)
import pytesseract

def main():
    src = cv2.imread('image.jpg')

    regions = detect_text(src)
    for x, y, w, h in regions:
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', src)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
