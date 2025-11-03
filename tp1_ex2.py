import cv2 as cv
import numpy as np

# marges d'erreurs en H et en S
eps_h = 40
eps_s = 80

kernel = np.ones((5, 5), np.uint8)

img = cv.imread("P:/MEA4/Perception_1/balle_small.jpg")
if img is None:
    print("Erreur lors du chargement de l'image")
    exit()

# conversion HSV
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Fonction pour récupérer la couleur du pixel sur lequel on clique
def get_pixel_value(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = img_hsv[y, x]
        param["color"] = np.array([h, s, v], dtype=np.uint8)
        print("Couleur pixel (HSV):", param["color"])

# Fonction pour créer le masque basé sur la couleur cliquée
def make_mask(img_hsv, color):
    h, s, v = int(color[0]), int(color[1]), int(color[2])
    lower = np.array([max(h - eps_h, 0), max(s - eps_s, 0), 0])
    upper = np.array([min(h + eps_h, 179), min(s + eps_s, 255), 255])
    mask = cv.inRange(img_hsv, lower, upper)
    return mask

# Fonction d'ouverture
def Ouverture(mask):
    mask_eroded = cv.erode(mask, kernel, 2)
    mask_open = cv.dilate(mask_eroded, kernel, 10)
    return mask_open

# Fonction principale
data = {"img": img.copy(), "color": None}
while(1):
    cv.imshow("image", img)
    cv.setMouseCallback("image", get_pixel_value, param=data)

    if data["color"] is not None:
        mask = make_mask(img_hsv, data["color"])
        mask_open = Ouverture(mask)
        img_highlight = img.copy()
        img_highlight[mask > 0] = [0, 255, 0]

        # flou gaussien, permet de réduire le bruit
        mask_blurred = cv.GaussianBlur(mask_open, (15, 15), 0)

        # détection des cercles
        mask_gray = mask_blurred 
        rows = mask_gray.shape[0]
        
        # on utilise la méthode de détection de Hough
        circles = cv.HoughCircles(mask_gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=100, param2=30, minRadius=10, maxRadius=280)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])  # Centre du cercle
                radius = i[2]  # Rayon du cercle
                # Dessiner le centre du cercle
                cv.circle(img_highlight, center, 1, (0, 100, 100), 3)
                # Dessiner le contour du cercle
                cv.circle(img_highlight, center, radius, (255, 0, 255), 3)
        else:
            print("Pas de cercle détecté")

        cv.imshow("detected circles", img_highlight)

    
    key = cv.waitKey(5) & 0xFF
    if key == ord('q'):  # q pour quitter
        break

cv.destroyAllWindows()
