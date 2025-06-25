#GERAR MALHA
import numpy as np

def gerar_malha(x1, y1, x2, y2, R, mapa):
    """
    Gera uma malha hexagonal a partir dos pontos (x1, y1) e (x2, y2) com raio R,
    cobrindo a Área A (mapa == 1).
    """
    coords = []
    max_x, max_y = mapa.shape

    
    # Vetor base (direção principal)
    dx = x2 - x1
    dy = y2 - y1
    distancia_base = np.sqrt(dx**2 + dy**2)

    if distancia_base == 0:
        print("Erro: (x1, y1) e (x2, y2) não podem ser iguais.")
        return []

    # Unidade do vetor base
    ux = dx / distancia_base
    uy = dy / distancia_base

    # Vetor perpendicular (para linhas)
    perp_ux = -uy
    perp_uy = ux

    # Distâncias entre centros
    dist_entre_circulos = R * np.sqrt(3)      # entre colunas
    dist_entre_linhas = (3 / 2) * R           # entre linhas
    fase = 0.5 * R * np.sqrt(3)               # deslocamento para colunas ímpares

    # Calcular alcance necessário para cobrir o mapa inteiro
    diag = np.sqrt(max_x**2 + max_y**2)
    max_linhas = int(diag / dist_entre_linhas) + 5
    max_colunas = int(diag / dist_entre_circulos) + 5

    for linha in range(-max_linhas, max_linhas + 1):
        offset_x = linha * dist_entre_linhas * perp_ux
        offset_y = linha * dist_entre_linhas * perp_uy
        fase_offset = (linha % 2) * fase

        for coluna in range(-max_colunas, max_colunas + 1):
            desloc_x = (coluna * dist_entre_circulos + fase_offset) * ux
            desloc_y = (coluna * dist_entre_circulos + fase_offset) * uy

            cx = x1 + desloc_x + offset_x
            cy = y1 + desloc_y + offset_y

            cx_int = int(round(cx))
            cy_int = int(round(cy))

            if 0 <= cx_int < max_x and 0 <= cy_int < max_y:
                if mapa[cx_int, cy_int] == 1:
                    coords.append((cx_int, cy_int))
    
    return coords
