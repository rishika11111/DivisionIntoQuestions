import cv2
import numpy as np
image = cv2.imread('img.png')
alpha = 2.0
beta = -160
new = alpha * image + beta
image = np.clip(new, 0, 255).astype(np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (29,837), 0)
thresh = cv2.threshold(blur, 0, 245, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
rows, cols = thresh.shape
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))

dilate = cv2.dilate(thresh, kernel, iterations=10)
count=0
# Find contours and draw rectangle
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if(((y+h)-y)==rows):
        count+=1
width = cols//count
for i in range(count):
    x1 = (i)*width
    x2 = (i+1)*width
    blur = cv2.GaussianBlur(gray[:,x1:x2], (27,33), 0)
    thresh = cv2.threshold(blur, 0, 245, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    rows, cols = thresh.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
    dilate = cv2.dilate(thresh, kernel, iterations=10)
    count=0
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if (w>=width-150 and h>=50):
            cv2.rectangle(image, (x+x1, y), (x+x1+ w, y + h), (36,255,12), 2)
cv2.imshow('image', image)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
