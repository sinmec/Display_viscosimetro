import cv2
import os
import time
from datetime import datetime

pasta_destino = "capturas_29_07_3"

if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)

camera = cv2.VideoCapture(0)

time.sleep(0)

print("Iniciando captura de frames a cada 1 segundos. Pressione Ctrl+C para parar.")

try:
    contador = 1
    while True:
        ret, frame = camera.read()

        if ret:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            nome_arquivo = f"frame_{timestamp}.jpg"
            caminho = os.path.join(pasta_destino, nome_arquivo)
            cv2.imwrite(caminho, frame)
            print(f"[{contador}] Frame salvo em: {caminho}")
        else:
            print(f"[{contador}] Erro ao capturar o frame.")

        contador += 1
        time.sleep(1)

except KeyboardInterrupt:
    print("\nCaptura interrompida pelo usuário.")

finally:
    camera.release()
    print("Câmera desligada.")
