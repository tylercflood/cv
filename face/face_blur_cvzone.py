import cv2
from cvzone.FaceDetectionModule import FaceDetector

cap = cv2.VideoCapture(0)

cap.set(3, 640) # width 
cap.set(4, 480) # height

detector = FaceDetector(minDetectionCon=0.9) # confidence level

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img, draw=True)

    if bboxs:
        for i, bbox in enumerate(bboxs):
            x,y,w,h = bbox['bbox']
            if x < 0: x = 0
            if y <0: y = 0
            imgCrop = img[y:y+h,x:x+w]
            imgBlur = cv2.blur(imgCrop, (35, 35))
            img[y:y+h,x:x+w] = imgBlur
            # cv2.imshow(f'Image Cropped {i}', imgCrop)

    
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()