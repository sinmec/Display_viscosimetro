import cv2
import numpy as np

matrizes_referencia = {
    '0': [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,1,1],[1,0,1,0,1],[1,1,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
    '1': [[0,0,1,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]],
    '2': [[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,1,1,1,1]],
    '3': [[1,1,1,1,1],[0,0,0,1,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
    '4': [[0,0,0,1,0],[0,0,1,1,0],[0,1,0,1,0],[1,0,0,1,0],[1,1,1,1,1],[0,0,0,1,0],[0,0,0,1,0]],
    '5': [[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[0,0,0,0,1],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
    '6': [[0,0,1,1,0],[0,1,0,0,0],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
    '7': [[1,1,1,1,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0]],
    '8': [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
    '9': [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,1],[0,0,0,0,1],[0,0,0,1,0],[0,1,1,0,0]],
    'vazio': [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
    '.': [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,1,1,0,0],[0,1,1,0,0]]
}

regioes = ((18, 24, 18, 22), (18, 24, 24, 28), (18, 24, 30, 34),
           (18, 24, 36, 40), (18, 24, 42, 46), (18, 24, 48, 52))

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ret, frame = cam.read()
cam.release()

if not ret:
    exit()

# # teste do codigo pra uma imagem do computador
# imagem_path = "pasta_frames_2025-08-05_16/frame_2025-08-05_16-02-54.jpg"
# frame = cv2.imread(imagem_path)
# if frame is None:
#     print("Erro: não foi possível abrir a imagem.")
#     exit()

img_blue = frame[:,:,0]
img_blue[img_blue > 100] = 255
img_blue[img_blue <= 100] = 0
contour = max(cv2.findContours(img_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
rect = cv2.minAreaRect(contour)
(w, h), angle = rect[1], rect[2]
if w < h: w, h, angle = h, w, angle - 90
M = cv2.getRotationMatrix2D(rect[0], angle, 1.0)
rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
box = cv2.boxPoints(((rect[0][0], rect[0][1]), (w, h), 0))
box = np.intp(cv2.transform(np.array([box]), M)[0])
x, y, w, h = cv2.boundingRect(box)
crop = rotated[y:y+h, x:x+w][82:h-99, 573:w-419]

img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
_, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
altura_celula, largura_celula = img_bin.shape[0] / 34, img_bin.shape[1] / 59
matriz = [[1 if np.sum(img_bin[int(i*altura_celula):int((i+1)*altura_celula),
                                int(j*largura_celula):int((j+1)*largura_celula)] == 255) /
                 (altura_celula*largura_celula) > 0.4 else 0
           for j in range(59)] for i in range(34)]

def submatriz(m, li, lf, ci, cf): return [linha[ci:cf+1] for linha in m[li:lf+1]]
def compara(a, b): return all(a[i][j] == b[i][j] for i in range(len(a)) for j in range(len(a[0])))
def reconhece(m):
    for simb, ref in matrizes_referencia.items():
        if compara(m, ref): return simb
    return "?"

valor_str = "".join(reconhece(submatriz(matriz, *r)).replace("vazio", "0") for r in regioes)
try:
    viscosidade = float(valor_str)
    print(f"{viscosidade} cP")
except:
    print(f"{valor_str}")

# # DEBUG: imagem rotacionada, cortada e binarizada em preto e branco.
# cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
# cv2.imshow('frame', frame)
# cv2.waitKey(0)
# cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
# cv2.imshow('frame', img_bin)
# cv2.waitKey(0)