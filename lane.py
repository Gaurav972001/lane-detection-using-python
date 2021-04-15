import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(grey, 50, 150)
    return canny

def region(image):
    height=image.shape[0]
    polygons=np.array([[(200,height),(1100,height),(550,250)]])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_img=cv2.bitwise_and(image,mask)
    return masked_img

def display_lines(image, lines):
    lines_img=np.zeros_like(image)
    if lines is not None:
        for  x1,y1,x2,y2 in lines:
            cv2.line(lines_img,(x1,y1),(x2,y2),(0,0,255),9)
    return lines_img

def make_coordinates(image,line_parameters):
    slope, intercept=line_parameters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def avg_slope_intercept(image, lines ):
    left_fit=[]
    right_fit=[]
    for line in lines :
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_avg=np.average(left_fit,axis=0)
    right_fit_avg=np.average(right_fit,axis=0)
    left_line=make_coordinates(image,left_fit_avg)
    right_line=make_coordinates(image, right_fit_avg)
    return np.array([left_line, right_line])



cap=cv2.VideoCapture("lanevid.mp4")
while(cap.isOpened()):
    _,frame=cap.read()
    canny_img = canny(frame)
    cropped_image = region(canny_img)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    avg_lines = avg_slope_intercept(frame, lines)
    lines_img = display_lines(frame, avg_lines)
    combo_image = cv2.addWeighted(frame, 0.76, lines_img, 1.2, 0)

    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
