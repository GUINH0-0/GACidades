import numpy as np
import matplotlib.pyplot as plt
import random

# === Parte 1: Leitura do mapa ===
def carregar_mapa(caminho):
    return np.loadtxt(caminho, dtype=int)

# === Parte 2: Geração da máscara circular ===
def gerar_mascara_circular(raio):
    tamanho = 2 * raio + 1
    mascara = np.zeros((tamanho, tamanho), dtype=int)
    for i in range(tamanho):
        for j in range(tamanho):
            dx = i - raio
            dy = j - raio
            if np.sqrt(dx**2 + dy**2) <= raio:
                mascara[i][j] = 1
    return mascara

# === Parte 3: Aplicar máscara nos centros ===
def aplicar_mascara(mapa_original, centros, mascara):
    mapa = mapa_original.copy()
    raio = mascara.shape[0] // 2

    for centro_x, centro_y in centros:
        for i in range(mascara.shape[0]):
            for j in range(mascara.shape[1]):
                if mascara[i][j] == 1:
                    x = centro_x + i - raio
                    y = centro_y + j - raio
                    if 0 <= x < mapa.shape[0] and 0 <= y < mapa.shape[1]:
                        if mapa[x][y] == 1:
                            mapa[x][y] = 2
                        elif mapa[x][y] == 2:
                            mapa[x][y] = 3  # sobreposição
    return mapa

# === Parte 5: Cálculo do fitness ===
def calcular_fitness(num_circulos, mapa_original, centros, mascara, alpha=100, beta=1):
    mapa_coberto = aplicar_mascara(mapa_original, centros, mascara)
    total_para_cobrir = np.count_nonzero(mapa_original == 1)
    total_coberto = np.count_nonzero(mapa_coberto == 2)

    porcentagem_coberta = total_coberto / total_para_cobrir if total_para_cobrir > 0 else 0
    fitness = alpha * porcentagem_coberta - beta * len(centros)

    return fitness, mapa_coberto

# === Parte 6: Visualização ===
import matplotlib.colors as mcolors

def visualizar_mapa(mapa, centros=None, raio=None, rota=None):
    plt.figure(figsize=(10, 10))

    # Define cores para valores: 0=fundo, 1=área a cobrir, 2=coberta, 3=sobreposição
    cores = ['black', 'lightgrey', 'white', 'lightcyan']
    cmap = mcolors.ListedColormap(cores)

    bounds = [0, 0.5, 1.5, 2.5, 3.5]  # para separar as classes
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(mapa, cmap=cmap, norm=norm)

    # Plota os círculos da cobertura
    if centros is not None and raio is not None:
        for centro_x, centro_y in centros:
            circulo = plt.Circle((centro_y, centro_x), raio, color='blue', fill=False, linewidth=1.5)
            plt.gca().add_patch(circulo)

    # Plota a rota TSP
    if rota is not None and centros is not None:
        rota_coords = [centros[i] for i in rota]
        rota_x, rota_y = zip(*rota_coords)
        plt.plot(rota_y, rota_x, color='red', linewidth=2, marker='o')  # Perceba: inverte y e x pra plotagem correta

    plt.title('Cobertura Gerada (com sobreposição azul claro e rota TSP em vermelho)')
    plt.axis('off')
    plt.show()