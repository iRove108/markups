import cv2
import numpy as np

from .textdetection import detect_text

class HighlightExtractor():
    """
    Class used to extract highlighted regions from an image
    """
    def __init__(self, lowerb = np.array([23, 50, 50]), upperb = np.array([40, 255, 255])):
        """
        Initialize highlight extractor with upper and lower HSV bounds to filter out highlight color
        Args:
            lowerb: Lower HSV bound
            upperb: Upper HSV bound
        Returns:
            None
        """
        # decent bounds for yellow highlighting
        self.lowerb = lowerb
        self.upperb = upperb

    def set_bounds(self, upperb, lowerb):
        """
        Function to reset HSV bounds of a HighlightExtractor
        Args:
            lowerb: Lower HSV bound
            upperb: Upper HSV bound
        Returns:
            None
        """
        self.lowerb = lowerb
        self.upperb = upperb

    def extract(self, src):
        """
        Extract highlighted regions of a BGR image based on upper and lower HSV bounds

        Args:
            src (np.array): BGR source image

        Returns:
            np.array: The masked image
        """
        src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(src_hsv, self.lowerb, self.upperb)

        # Remove isolated regions of mask
        kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (7, 7))
        cv2.erode(mask, kernel, mask)
        cv2.dilate(mask, kernel, mask)

        # Tune the mask with text detection;
        # add text region to mask if > 51% is highlighted (i.e. present in current mask)
        text_bboxes = detect_text(src)
        for x, y, w, h in text_bboxes:
            text_mask = np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)
            text_mask[y:y+h, x:x+w] = 255 # mark bbox region white in mask

            ratio_highlighted = cv2.countNonZero(mask & text_mask) / (w * h)
            if ratio_highlighted > .5:
                mask |= text_mask

        return mask

    def sat_extract(self, src):
        """
        Extract highlighted regions of a BGR image based on areas of high saturation

        Args:
            src (np.array): BGR source image

        Returns:
            np.array: The masked image
        """
        src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        mask = src_hsv[:, :, 1]
        mask = cv2.bilateralFilter(mask,25,75,75)
        trash, mask = cv2.threshold(mask,50,255,cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask = cv2.dilate(mask,kernel,iterations = 1)
        return mask
