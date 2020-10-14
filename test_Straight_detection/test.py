import cv2
import numpy as np

height = 500
width = 800
img = np.zeros((height,width,3),np.uint8)
cv2.putText(img,"COIVD",(10,300),cv2.FONT_HERSHEY_COMPLEX,8,(255,255,255),30)
img = cv2.bitwise_not(img)
cv2.imwrite('covid.jpg',img)
cv2.imshow("img",img)
cv2.waitKey(0)