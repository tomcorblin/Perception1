import cv2 as cv
import numpy as np

video_path = "C:/Users/thomas.corblin/Downloads/balle.mp4"
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur d'ouverture")
    exit()

while True:
    # decoupage en frame
    ret, frame = cap.read()

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    if not ret:
        print("Impossible de récupérer une frame (fin du flux ?). Sortie...")
        break

    # Traitement d'image sur chaque frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (9, 9), 2)

    morph_open = cv.morphologyEx(blurred, cv.MORPH_OPEN, kernel)
    morph_close = cv.morphologyEx(morph_open, cv.MORPH_CLOSE, kernel)

    circles = cv.HoughCircles(morph_close, cv.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=50, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # On selectionne le premier cercle trouvé
        x, y, r = circles[0, 0]
        cv.circle(frame, (x, y), r, (0, 255, 0), 3)
        cv.circle(frame, (x, y), 2, (0, 0, 255), 3)

    # Affichage de l'image traitée
    cv.imshow('Frame', frame)
    key = cv.waitKey(20) & 0xFF
    if key == ord('q'):  # q pour quitter
        break

# Libération de la capture et fermeture des fenêtres
cap.release()
cv.destroyAllWindows()
