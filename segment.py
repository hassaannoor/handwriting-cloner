import cv2
import numpy as np

def segment_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    dilated = cv2.dilate(bw, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for c in sorted(contours, key=lambda x: cv2.boundingRect(x)[1]):
        x,y,w,h = cv2.boundingRect(c)
        lines.append(image[y:y+h, x:x+w])

    return lines
