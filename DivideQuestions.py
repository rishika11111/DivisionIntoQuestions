import cv2
import numpy as np

def NumberOfPartitions(gray):
    blur = cv2.GaussianBlur(gray, (29,837), 0)
    thresh = cv2.threshold(blur, 0, 245, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    rows, cols = thresh.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))

    dilate = cv2.dilate(thresh, kernel, iterations=10)
    count=0
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if(((y+h)-y)==rows):
            count+=1
    return count
def res():
    while True:
        image = cv2.imread('img.png')
        #Removing watermark in order to divide without its involvement.
        alpha = 2.0
        beta = -160
        new = alpha * image + beta
        image = np.clip(new, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (35,37), 0)
        thresh = cv2.threshold(blur, 0, 245, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        rows, cols = thresh.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))
        dilate = cv2.dilate(thresh, kernel, iterations=10)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = max(cnts, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        count = NumberOfPartitions(gray)
        minwidth = cols//count
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if (w>=minwidth-150 and h>=50): cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.imshow('image', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
res()
