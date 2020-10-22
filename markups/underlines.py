import cv2
import numpy as np

class UnderlineExtractor():
    """
    Class used to extract underlined text from an image
    """
    def __init__(self, cannyb = np.array([80, 120]), min_line_len = 30, max_line_gap = 1)):
        """
        Initialize underline extractor with canny threshold bounds for edge detection
        and arguments for houghlines
        Args:
            cannyb: np array containing the weak and strong thresholds for canny
            min_line_len: Minimum line length. Line segments shorter than that are rejected
            max_line_gap: Maximum allowed gap between points on the same line to link them
        Returns:
            None
        """
        self.cannyb = cannyb
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap

    def set_bounds(self, cannyb=None, min_line_len=None, max_line_gap=None):
        """
        Function to reset bounds of an UnderlineExtractor
        Args:
            cannyb: np array containing the weak and strong thresholds for canny
            min_line_len: Minimum line length. Line segments shorter than that are rejected
            max_line_gap: Maximum allowed gap between points on the same line to link them
        Returns:
            None
        """
        if cannyb is not None:
            self.cannyb = cannyb
        if min_line_len is not None:
            self.min_line_len = min_line_len
        if max_line_gap is not None:
            self.max_line_gap = max_line_gap

    def extract(self, src):
        """
        Extract underlined regions of a BGR image based on canny edge detection and houghlines

        Args:
            src (np.array): BGR source image

        Returns:
            np.array: The image with all horizontal lines marked
        """

	gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, self.cannyb[0], self.cannyb[1])
	lines = cv2.HoughLinesP(edges, 1, math.pi/2, 2, None, self.min_line_len, self.max_line_gap);
	for line in lines[0]:
   	    pt1 = (line[0],line[1])
    	    pt2 = (line[2],line[3])
    	    cv2.line(src, pt1, pt2, (0,0,255), 3)

        #TODO what should we do from here once we have the lines marked?

        return src

