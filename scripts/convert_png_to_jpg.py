# Run in Conda base environment
import cv2
import glob
import os
os.chdir("C:/Users/Emma.Wu/Documents/Hackathon/images/samples")

i=0

images = glob.glob("*.png")

for i in images:
    print("start")
    img = cv2.imread(i, 1)  
    cv2.imwrite(i.split('.')[0]+'.jpg', img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    print("end")
