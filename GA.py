#GA COM MALHA

import numpy as np
from ga import gago
from utilities import bits2bytes
import matplotlib.pyplot as plt
from functools import partial
from acharota import genetic_tsp
from geramalha import gerar_malha
import funcs as f
import subprocess
import sys
sys.path.append(r'C:\Users\avanc\Desktop\trabalho_py-main')

# Função de fitness usando remapeamento por módulo em vez de clip
def fit_func(bits, mapa_original, mascara, R):
    X = bits2bytes(bits, 'uint8').astype(int)

    # Garante que temos pelo menos 4 variáveis
    if len(X) < 4:
        return 1e9

    # Limita os valores para o intervalo desejado [1, 128]
    x1 = max(1, min(128, X[0]))
    y1 = max(1, min(128, X[1]))
    x2 = max(1, min(128, X[2]))
    y2 = max(1, min(128, X[3]))

    coords_malha = gerar_malha(x1, y1, x2, y2, R, mapa_original)

    if len(coords_malha) == 0:
        return 1e9

    mapa_coberto = f.aplicar_mascara(mapa_original, coords_malha, mascara)

    total_para_cobrir = np.count_nonzero(mapa_original == 1)
    total_coberto = np.count_nonzero((mapa_coberto == 2) | (mapa_coberto == 3))
    sobreposicoes = np.count_nonzero(mapa_coberto == 3)

    faltando = total_para_cobrir - total_coberto
    erro = faltando + len(coords_malha)  # Penaliza por número de círculos usados

    return erro


# Função principal
def main():
    caminho_mapa = "mapa.dat"
    raio = 50

    mapa = f.carregar_mapa(caminho_mapa)
    mascara = f.gerar_mascara_circular(raio)

    gaoptions = {
        "PopulationSize": 10,
        "Generations": 10,
        "InitialPopulation": [],
        "MutationFcn": 0.15,
        "EliteCount": 2,
    }


    fit = partial(fit_func, mapa_original=mapa, mascara=mascara, R=raio)
    result = gago(fit, 32, gaoptions)


    # Extrai a melhor solução
    bits_optimizado = result[0]

    # Decodifica os bits para inteiros no intervalo [1, 128]
    X = bits2bytes(bits_optimizado, 'uint8').astype(int)
    x1 = max(1, min(128, X[0]))
    y1 = max(1, min(128, X[1]))
    x2 = max(1, min(128, X[2]))
    y2 = max(1, min(128, X[3]))

    print(f"x1, y1: ({x1}, {y1})")
    print(f"x2, y2: ({x2}, {y2})")

    # Gera a malha com os melhores parâmetros
    centros = gerar_malha(x1, y1, x2, y2, raio, mapa)
    rota, distancia = genetic_tsp(centros, 500)

    print(f"Centros válidos: {len(centros)}")


    # Aplica a máscara e visualiza
    mapa_coberto = f.aplicar_mascara(mapa, centros, mascara)

    total_para_cobrir = np.count_nonzero(mapa == 1)
    total_coberto = np.count_nonzero((mapa_coberto == 2) | (mapa_coberto == 3))
    faltando = total_para_cobrir - total_coberto
    print("Total para cobrir: ", total_para_cobrir)
    print("Total coberto: ", total_coberto)
    print("Faltando: ",faltando)

    f.visualizar_mapa(mapa_coberto, centros, raio)

if __name__ == "__main__":
    main()
#