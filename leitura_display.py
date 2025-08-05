import cv2
import numpy as np
import os
import time
from datetime import datetime
import csv
from glob import glob

timestamp = datetime.now().strftime("%Y-%m-%d_%H") # %Y-%m-%d_%H-%M-%S"
pasta_destino = f"capturas_processadas_{timestamp}"
arquivo_csv = f"viscosidade_{timestamp}.lvm"
pasta_frames = f"pasta_frames_{timestamp}"

if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)
if not os.path.exists(pasta_frames):
    os.makedirs(pasta_frames)

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

def processar_imagem(image):
    # image = cv2.imread(imagem_path)
    image_blue = image[:,:,0]
    image_blue[image_blue > 100] = 255
    image_blue[image_blue <= 100] = 0

    # #DEBUG: verificar contornos
    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cv2.imshow('frame', image_blue)
    # cv2.waitKey(0)

    contours, _ = cv2.findContours(image_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Nenhum contorno encontrado.")
    contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(contour)
    (w, h) = rect[1]
    angle = rect[2]
    if w < h:
        w, h = h, w
        angle -= 90

    # # DEBUG: imagem de entrada e linha de contorno
    # image_blue_debug = cv2.cvtColor(image_blue, cv2.COLOR_GRAY2BGR)
    # box = cv2.boxPoints(rect)
    # cv2.drawContours(image_blue_debug, [box.astype(int)], 0, (0, 0, 255), 2)
    # out_image = np.vstack((image, image_blue_debug))
    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cv2.imshow('frame', out_image)
    # cv2.waitKey(0)

    center = tuple(map(float, rect[0]))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    box = cv2.boxPoints(((center[0], center[1]), (w, h), 0))
    box = np.intp(cv2.transform(np.array([box]), M)[0])
    x, y, w, h = cv2.boundingRect(box)
    warped = rotated[y:y+h, x:x+w]

    cropped = warped[78:h-96, 575:w-417]
    # cropped = warped[90:h - 87, 507:w - 367]

    return cropped

def salvar_csv(tempo, viscosidade, caminho_csv):
    # header = ['timestamp', 'viscosidade']
    header = ['tempo_em_segundos', 'viscosidade']
    escrever_header = not os.path.exists(caminho_csv)
    with open(caminho_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if escrever_header:
            writer.writerow(header)
        writer.writerow([tempo, viscosidade])

def carregar_e_converter_para_matriz(img, linhas=34, colunas=59, percentual_min=0.4):#path_imagem no lugar de img
    # img = cv2.imread(path_imagem)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # DEBUG: imagem rotacionada, cortada e binarizada em preto e branco.
    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cv2.imshow('frame', img_bin)
    # cv2.waitKey(0)

    altura, largura = img_bin.shape
    altura_celula = altura / linhas
    largura_celula = largura / colunas

    matriz_binaria = []
    for i in range(linhas):
        linha = []
        for j in range(colunas):
            y1 = int(i * altura_celula)
            y2 = int((i + 1) * altura_celula)
            x1 = int(j * largura_celula)
            x2 = int((j + 1) * largura_celula)
            celula = img_bin[y1:y2, x1:x2]
            brancos = np.sum(celula == 255)
            proporcao = brancos / celula.size
            linha.append(1 if proporcao > percentual_min else 0)
        matriz_binaria.append(linha)

    return matriz_binaria

def comparar_matrizes(m1, m2):
    total = sum(len(linha) for linha in m1)
    iguais = sum(1 for i in range(len(m1)) for j in range(len(m1[0])) if m1[i][j] == m2[i][j])
    return (iguais / total) * 100 >= 100

def identificar_simbolo(matriz_entrada):
    for simbolo, matriz_ref in matrizes_referencia.items():
        if comparar_matrizes(matriz_entrada, matriz_ref):
            return simbolo
    return None

def extrair_submatriz(matriz, li, lf, ci, cf):
    return [linha[ci:cf+1] for linha in matriz[li:lf+1]]

def identificar_sequencia(matriz, regioes):
    simbolos = []
    for regiao in regioes:
        submatriz = extrair_submatriz(matriz, *regiao)
        simbolo = identificar_simbolo(submatriz)
        simbolos.append(simbolo if simbolo else "?")
    return simbolos

camera = cv2.VideoCapture(0)
print("Capturando imagens a cada 1 segundo. Ctrl+C para parar.")

regioes = [(18, 24, 18, 22), (18, 24, 24, 28), (18, 24, 30, 34),
           (18, 24, 36, 40), (18, 24, 42, 46), (18, 24, 48, 52)]

try:
    contador_segundos = 0
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Erro ao capturar imagem.")
            continue

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        caminho_frame = os.path.join(pasta_frames, f"frame_{timestamp}.jpg")
        cv2.imwrite(caminho_frame, frame)

        try:
            imagem_processada = processar_imagem(frame)
            imagem_caminho = os.path.join(pasta_destino, f"processado_{timestamp}.jpg")
            cv2.imwrite(imagem_caminho, imagem_processada)

            matriz_bin = carregar_e_converter_para_matriz(imagem_processada)
            simbolos = identificar_sequencia(matriz_bin, regioes)
            valor_str = "".join(s.replace("vazio", "0") for s in simbolos)

            try:
                 viscosidade = float(valor_str)
                 print(f"[{contador_segundos}][{timestamp}]  Viscosidade identificada: {viscosidade} cP")
            except ValueError:
                 print(f"[{timestamp}] Valor inválido lido: {valor_str}")
                 viscosidade = float('nan')

            salvar_csv(contador_segundos, viscosidade, arquivo_csv)
            contador_segundos += 1

        except Exception as e:
            print(f"[{timestamp}] Erro ao processar imagem: {e}")

        time.sleep(1)

except KeyboardInterrupt:
    print("\nCaptura encerrada.")

finally:
    camera.release()
    print("Câmera desligada.")

# # Bloco de teste para imagens em uma pasta
# imagens = sorted(glob("capturas_05_08/*.jpg"))
# contador_segundos = 1
# for caminho_frame in imagens:
#     timestamp = os.path.basename(caminho_frame).replace("frame_", "").replace(".jpg", "")
#
#     try:
#         imagem_processada = processar_imagem(caminho_frame)
#         # imagem_caminho = os.path.join(pasta_destino, f"processado_{timestamp}.jpg")
#         # cv2.imwrite(imagem_caminho, imagem_processada)
#
#         matriz_bin = carregar_e_converter_para_matriz(imagem_processada)
#         simbolos = identificar_sequencia(matriz_bin, regioes)
#         valor_str = "".join(s.replace("vazio", "0") for s in simbolos)
#
#         try:
#             viscosidade = float(valor_str)
#             print(f"[{timestamp}] Viscosidade identificada: {viscosidade} cP")
#         except ValueError:
#             print(f"[{timestamp}] Valor inválido lido: {valor_str}")
#             viscosidade = float('nan')
#
#         salvar_csv(contador_segundos, viscosidade, arquivo_csv)
#         contador_segundos += 1
#
#     except Exception as e:
#         print(f"[{timestamp}] Erro ao processar imagem: {e}")