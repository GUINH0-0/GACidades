#ACHA ROTA

import numpy as np
import random
import matplotlib.pyplot as plt

def genetic_tsp(coordinates, generations, population_size=100, crossover_rate=0.9, mutation_rate=0.1, elitism=False):
    coordinates = np.array(coordinates)
    num_cities = len(coordinates)

    # Calcula a matriz de distâncias
    def euclidean_distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    dist_matrix = np.array([
        [euclidean_distance(coordinates[i], coordinates[j]) for j in range(num_cities)]
        for i in range(num_cities)
    ])

    # Função para calcular distância total de uma rota
    def calculate_distance(route):
        return sum(dist_matrix[route[i], route[i + 1]] for i in range(num_cities - 1)) + dist_matrix[route[-1], route[0]]

    # Geração inicial
    def generate_population():
        return np.array([np.random.permutation(num_cities) for _ in range(population_size)])

    # Seleção por torneio
    def tournament_selection(population):
        i1, i2 = random.sample(range(population_size), 2)
        return population[i1] if calculate_distance(population[i1]) < calculate_distance(population[i2]) else population[i2]

    # Crossover por ordem
    def order_crossover(parent1, parent2):
        start, end = sorted(random.sample(range(num_cities), 2))
        child = [-1] * num_cities
        child[start:end] = parent1[start:end]
        remaining = [gene for gene in parent2 if gene not in child]
        j = 0
        for i in range(num_cities):
            if child[i] == -1:
                child[i] = remaining[j]
                j += 1
        return np.array(child)

    # Mutação simples
    def mutate(route):
        if random.random() < mutation_rate:
            i, j = random.sample(range(num_cities), 2)
            route[i], route[j] = route[j], route[i]
        return route

    # Algoritmo genético principal
    population = generate_population()
    best_route = min(population, key=calculate_distance)
    best_distance = calculate_distance(best_route)

    for gen in range(generations):
        new_population = []

        if elitism:
            new_population.append(best_route)

        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            if random.random() < crossover_rate:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1.copy()
            child = mutate(child)
            new_population.append(child)

        population = np.array(new_population)
        current_best = min(population, key=calculate_distance)
        current_best_distance = calculate_distance(current_best)

        if current_best_distance < best_distance:
            best_route = current_best
            best_distance = current_best_distance

        if gen % 100 == 0:
            print(f"Geração {gen} | Melhor distância: {best_distance:.2f}")

    return best_route, best_distance


