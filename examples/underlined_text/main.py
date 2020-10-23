import cv2
import numpy as np
from markups import (
    UnderlineExtractor    
)

def main():
    src = cv2.imread('image.jpg')
    src_small = cv2.resize(src, (0,0), fx=0.5, fy=0.5)

    extractor = UnderlineExtractor(cannyb=np.array([200, 300]), min_line_len=200, max_line_gap=15)
    mask = extractor.extract(src_small)

    cv2.imshow('image', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
