import cv2
import numpy as np

cameraCapture = cv2.VideoCapture(1)

success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1:
    # Adicionar: reducao do tamanho do video? menos pixels para processar

    # LPF: remove noise e borra imagens
    # HPF: encontra bordas
    # Medianblur reduz noise enquanto mantem bordas, 
    # no entanto PODE ser mais um pouco lento que GaussianBlur
    blurredFrame = cv2.medianBlur(frame, 5)

    # Reforca contornos
    blurredFrame = cv2.Laplacian(blurredFrame, cv2.CV_8U)
    blurredFrame = cv2.subtract(frame, blurredFrame)

    # Converte imagem para HSV (mudanca de colorspace)
    hsvFrame = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)

    # Filtra cores
    maskFrame = cv2.inRange(hsvFrame, (10, 140, 10), (25, 255, 255))
    elEst = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    maskFrame = cv2.erode(maskFrame, elEst, iterations=2)
    elEst = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    maskFrame = cv2.dilate(maskFrame, None, iterations=2)

    # Find and draw contours
    contours, __ = cv2.findContours(maskFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (255,145,0), 10)

    # Transmite imagem ao vivo
    cv2.imshow("camera", frame)
    cv2.imshow("bluredFrame", blurredFrame)
    cv2.imshow("maskFrame", maskFrame)

    success, frame = cameraCapture.read()

cv2.destroyAllWindows()

# MedianBlur começa a ficar caro com valores altos de ksize, como 7. De repente usar 5
# blurredSrc = cv2.medianBlur(src, blurKseize)