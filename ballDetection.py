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
    cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

    # Essa parte ainda tenho que descobrir como funciona
    contours_poly = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])


    # drawing = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    
    for i in range(len(contours)):
        # talvez trocar o valor int(radius[i]) por um valor fixo (como 24, nesse caso)
        cv2.circle(frame, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), (0, 255, 0), 2)
    
    if contours:
        frame = cv2.putText(frame, f"({int(centers[0][0])}, {int(centers[0][1])})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)

    # Transmite imagem ao vivo
    cv2.imshow("camera", frame)
    cv2.imshow("bluredFrame", blurredFrame)
    cv2.imshow("maskFrame", maskFrame)

    success, frame = cameraCapture.read()

cv2.destroyAllWindows()

# MedianBlur começa a ficar caro com valores altos de ksize, como 7. De repente usar 5
# blurredSrc = cv2.medianBlur(src, blurKseize)