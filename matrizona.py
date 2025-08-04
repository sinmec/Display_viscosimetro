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
    '.': [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,1,1,0,0],[0,1,1,0,0]]}

def carregar_e_converter_para_matriz(path_imagem, linhas=34, colunas=59, percentual_min=0.4):
    img = cv2.imread(path_imagem)
    if img is None:
        raise FileNotFoundError(f"Imagem '{path_imagem}' não encontrada.")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

            ativa = 1 if proporcao > percentual_min else 0
            linha.append(ativa)
        matriz_binaria.append(linha)

    return matriz_binaria

def comparar_matrizes(m1, m2):
    total_elementos = 0
    elementos_iguais = 0

    for i in range(len(m1)):
        for j in range(len(m1[0])):
            total_elementos += 1
            if m1[i][j] == m2[i][j]:
                elementos_iguais += 1

    porcentagem_igualdade = (elementos_iguais / total_elementos) * 100
    return porcentagem_igualdade >= 100

def identificar_simbolo(matriz_entrada, referencias):
    for simbolo, matriz_ref in referencias.items():
        if comparar_matrizes(matriz_entrada, matriz_ref):
            return simbolo
    return None

def extrair_submatriz(matriz, lin_inicio, lin_fim, col_inicio, col_fim):
    return [linha[col_inicio:col_fim + 1] for linha in matriz[lin_inicio:lin_fim + 1]]

def identificar_sequencia_de_simbolos(matriz_entrada, referencias, regioes):
    simbolos = []

    for idx, regiao in enumerate(regioes):
        lin_ini, lin_fim, col_ini, col_fim = regiao
        submatriz = extrair_submatriz(matriz_entrada, lin_ini, lin_fim, col_ini, col_fim)

        # print(f"\nSubmatriz {idx + 1} (linhas {lin_ini}-{lin_fim}, colunas {col_ini}-{col_fim}):")
        # for linha in submatriz:
        #     print(" ".join(str(x) for x in linha))

        simbolo = identificar_simbolo(submatriz, referencias)
        simbolos.append(simbolo if simbolo else "?")

    return simbolos

if __name__ == "__main__":
    imagem = r"tela_azul_final.jpg"
    matriz_input = carregar_e_converter_para_matriz(imagem)

    # Regiões das submatrizes: (linha_ini, linha_fim, col_ini, col_fim)
    regioes = [(18, 24, 13, 17),(18, 24, 19, 23),(18, 24, 24, 28),(18, 24, 30, 34),(18, 24, 36, 40), (18, 24, 42, 46),(18, 24, 48, 52)]

    simbolos_identificados = identificar_sequencia_de_simbolos(matriz_input, matrizes_referencia, regioes)

    # print("\nMatriz obtida da imagem:")
    # for linha in matriz_input:
    #     print(" ".join(str(x) for x in linha))

    valor_str = "".join(simbolos_identificados)

    valor_str = valor_str.replace('vazio', '0')

    try:
        if '.' in valor_str:
            valor_viscosidade = float(valor_str)
        else:
            valor_viscosidade = float(valor_str)
        print(f"\nValor identificado para a viscosidade: {valor_viscosidade} cP")
    except ValueError:
        print(f"\nNão foi possível converter '{valor_str}' para número.")