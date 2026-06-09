"""
selection.py
Opérateurs de sélection pour l'algorithme génétique HDFS.

Deux stratégies :
  1. Roulette Wheel Selection (sélection proportionnelle à la fitness)
  2. Tournament Selection   (tournoi entre k individus)
"""

import random
from chromosome import Chromosome


# ======================================================================
# Roulette Wheel Selection
# ======================================================================

def roulette_wheel_selection(population: list, num_selected: int) -> list:
    """
    Sélection par roulette.
    La probabilité de sélection de l'individu i est :
        P(i) = fitness(i) / ∑_k fitness(k)

    Gestion des fitness négatives : décalage par min(fitness) + ε.

    Paramètres
    ----------
    population   : list[Chromosome]
    num_selected : int — Nombre de parents à sélectionner

    Retourne
    --------
    list[Chromosome] — Parents sélectionnés (avec remise)
    """
    # Normalisation pour gérer les valeurs négatives
    raw_fits = [c.fitness for c in population]
    min_fit  = min(raw_fits)
    shift    = abs(min_fit) + 1e-6 if min_fit <= 0 else 0
    adjusted = [f + shift for f in raw_fits]
    total    = sum(adjusted)

    if total == 0:
        return random.choices(population, k=num_selected)

    selected = []
    for _ in range(num_selected):
        pick = random.uniform(0, total)
        cumulative = 0.0
        chosen = population[-1]   # FIX: default to last individual to avoid
                                  # the for-else missing append on float drift
        for chrom, adj_fit in zip(population, adjusted):
            cumulative += adj_fit
            if cumulative >= pick:
                chosen = chrom
                break
        selected.append(chosen.clone())

    return selected


# ======================================================================
# Tournament Selection
# ======================================================================

def tournament_selection(population: list, num_selected: int,
                         tournament_size: int = 3) -> list:
    """
    Sélection par tournoi.
    Tire aléatoirement `tournament_size` individus et retourne le meilleur.

    Paramètres
    ----------
    population       : list[Chromosome]
    num_selected     : int — Nombre de parents à sélectionner
    tournament_size  : int — Taille du tournoi (k)

    Retourne
    --------
    list[Chromosome]
    """
    selected = []
    pop_size = len(population)

    for _ in range(num_selected):
        contestants = random.sample(population, min(tournament_size, pop_size))
        winner = max(contestants, key=lambda c: c.fitness)
        selected.append(winner.clone())

    return selected


# ======================================================================
# Sélection élitiste (gardée entre générations)
# ======================================================================

def elitism_selection(population: list, elite_size: int) -> list:
    """
    Retourne les `elite_size` meilleurs individus.
    Utilisé pour préserver l'élite d'une génération à l'autre.
    """
    sorted_pop = sorted(population, key=lambda c: c.fitness, reverse=True)
    return [c.clone() for c in sorted_pop[:elite_size]]