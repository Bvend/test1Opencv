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
    blurredFrame = cv2.GaussianBlur(frame, (11, 11), 0)

    # Converte imagem para HSV (mudanca de colorspace)
    hsvFrame = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)

    # Filtra cores
    maskFrame = cv2.inRange(hsvFrame, (10, 160, 10), (25, 255, 255))
    maskFrame = cv2.erode(maskFrame, None, iterations=2)
    maskFrame = cv2.dilate(maskFrame, None, iterations=2)

    # Transmite imagem ao vivo
    cv2.imshow("camera", frame)
    cv2.imshow("bluredFrame", blurredFrame)
    cv2.imshow("maskFrame", maskFrame)

    success, frame = cameraCapture.read()

cv2.destroyAllWindows()

# MedianBlur começa a ficar caro com valores altos de ksize, como 7. De repente usar 5
# blurredSrc = cv2.medianBlur(src, blurKseize)