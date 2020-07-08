import cv2
import numpy as np

#Function to remove headers and footers.
def boundary(gray):
    edges = cv2.Canny(gray,180,200,apertureSize = 3)
    minLineLength=75
    rows,cols,_= gray.shape
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100, minLineLength=100)
    linevalues=[]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if(y1!=y2): continue
        linevalues.append(y1)
    linevalues.sort()
    y1bound = max([y for y in linevalues if y<=rows//2])
    y2bound = min([y for y in linevalues if y>=rows//2])
    return y1bound,y2bound

def drawbound():
    image = cv2.imread('img.png')
    alpha = 2.0
    beta = -160
    new = alpha * image + beta
    image = np.clip(new, 0, 255).astype(np.uint8)
    image1 = image.copy()
    y1bound,y2bound = boundary(image)
    gray = cv2.cvtColor(image[y1bound:y2bound,:], cv2.COLOR_BGR2GRAY)
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
        x1 = (i)*width+cols//20
        x2 = (i+1)*width
        blur = cv2.GaussianBlur(gray[:,x1:x2], (29,31), 0)
        thresh = cv2.threshold(blur, 0, 245, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        rows, cols = thresh.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
        dilate = cv2.dilate(thresh, kernel, iterations=2)
        count=0
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if (w>=width-width//2):
                cv2.line(image,(x1,y+y1bound),(x2,y+y1bound), (255,0,0), 2)
                cv2.rectangle(image1, (x+x1, y+y1bound), (x+x1+w, y+y1bound + h), (36,255,12), 2)
        x,y,w,h = cv2.boundingRect(cnts[0])
    cv2.imshow('Lines separating questions ', image)
    cv2.imshow('Bounding Boxes', image1)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
drawbound()
