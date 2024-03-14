import cv2
import numpy as np

cap = cv2.VideoCapture(0)
bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=1000, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgMask = bgSubtractor.apply(frame)

    # trying t improve the mask
    fgMask = cv2.medianBlur(fgMask, 5)

    background_image = cv2.resize(cv2.imread('cv/face/assets/cozy.jpg'), (frame.shape[1], frame.shape[0]))

    # masked frame
    foreground = cv2.bitwise_and(frame, frame, mask=fgMask)
    invertedMask = cv2.bitwise_not(fgMask)
    background = cv2.bitwise_and(background_image, background_image, mask=invertedMask)
    combined = cv2.add(foreground, background)

    cv2.imshow('webcam', combined)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()