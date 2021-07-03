import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3 ,640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS,60)

segmentor = SelfiSegmentation(1)
fpsReader = cvzone.FPS()



listImg = os.listdir("Images")
print(listImg)
imgList = []
for imgPath in listImg :
    img = cv2.imread(f'Images/{imgPath}')
    imgList.append(img)
print(len(imgList))

imgIndex = 0
while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[imgIndex], threshold=0.8)


    imgStacked = cvzone.stackImages([img,imgOut],2,1)
    _, imgStacked = fpsReader.update(imgStacked,color=(0,0,255))


    cv2.imshow("Image",imgStacked)
    key = cv2.waitKey(1)
    print(imgIndex)
    if key == ord('a'):
        if imgIndex >0:
            imgIndex -=1
    elif key == ord('d'):
        if imgIndex<len(imgList)-1:
            imgIndex +=1
    elif key == ord('q'):
        break