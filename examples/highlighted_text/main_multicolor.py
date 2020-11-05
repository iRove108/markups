import cv2
import numpy as np
from markups import (
    HighlightExtractor,
)
# note: multicolor_highlights.jpg is from https://www.geocities.ws/pittinsky/page6.jpg
lower_bounds = {
    'pink':np.array([130, 20, 15]),
    'purple':np.array([100, 20, 15]),
    'blue':np.array([70, 20, 15]),
    'orange':np.array([10, 20, 15]),
}
upper_bounds = {
    'pink':np.array([179, 255, 255]),
    'purple':np.array([150, 255, 255]),
    'blue':np.array([110, 255, 255]),
    'orange':np.array([25, 255, 255]),
}
def main():
    src = cv2.imread('multicolor_highlights.jpg')
    src_small = cv2.resize(src, (0,0), fx=0.5, fy=0.5)

    color = 'orange'
    extractor = HighlightExtractor(lowerb = lower_bounds[color], upperb = upper_bounds[color])
    mask = extractor.extract(src_small)

    cv2.imshow('image', src_small & mask[:, :, np.newaxis])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
