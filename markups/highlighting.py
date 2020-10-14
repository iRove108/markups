import cv2
import numpy as np

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
        kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (5, 5))
        cv2.erode(mask, kernel, mask)
        cv2.dilate(mask, kernel, mask)

        return mask

