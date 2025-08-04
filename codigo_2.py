import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r'capturas_31_07/frame_2025-07-31.jpg'

image = cv2.imread(image_path)

image_blue = image[:,:,0]
image_blue[image_blue > 100] =255
image_blue[image_blue <= 100] =0

# cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
# cv2.imshow('frame', image_blue)
# cv2.waitKey(0)

contours, _ = cv2.findContours(image_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)

# cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
# cv2.imshow('frame', hsv)
# cv2.waitKey(0)

# Retângulo rotacionado
rect = cv2.minAreaRect(contour)
(w, h) = rect[1]
angle = rect[2]

if w < h:
    w, h = h, w
    angle += -90

image_blue_debug = cv2.cvtColor(image_blue, cv2.COLOR_GRAY2BGR)

box = cv2.boxPoints(rect)
cv2.drawContours(image_blue_debug,[box.astype(int)],0,(0,0,255),2)

out_image = np.vstack((image, image_blue_debug))

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.imshow('frame', out_image)
cv2.waitKey(1)

width, height = int(w), int(h)

center = (float(rect[0][0]), float(rect[0][1]))

M = cv2.getRotationMatrix2D(center, angle, 1.0)

rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

box = cv2.boxPoints(((center[0], center[1]), (w, h), 0))
box = np.intp(cv2.transform(np.array([box]), M)[0])
x, y, w, h = cv2.boundingRect(box)
warped = rotated[y:y+h, x:x+w]

cropped = warped[87:h-81, 504:w-360]
# cropped = warped[0:h-0, 0:w-0]
cv2.imwrite("tela_azul_final.jpg", cropped)

plt.figure(figsize=(10, 4))
plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("")
plt.show()
