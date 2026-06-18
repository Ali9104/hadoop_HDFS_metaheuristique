"""
crossover.py
Opérateurs de croisement pour l'algorithme génétique HDFS.

Deux variantes :
  1. Single-Point Crossover  (point de coupe unique)
  2. Uniform Crossover       (gène par gène, tirage aléatoire)
"""

import random
from chromosome import Chromosome


# ======================================================================
# Single-Point Crossover
# ======================================================================

def single_point_crossover(parent1: Chromosome, parent2: Chromosome,
                            crossover_rate: float = 0.8):
    """
    Croisement en un point.

    Exemple (n=5 blocs) :
        P1 : [N1 N2 N3 N1 N2]
        P2 : [N2 N1 N1 N3 N3]
        Point de coupe = 2
        E1 : [N1 N2 | N1 N3 N3]
        E2 : [N2 N1 | N3 N1 N2]

    Paramètres
    ----------
    parent1, parent2 : Chromosome
    crossover_rate   : float — Probabilité d'appliquer le croisement

    Retourne
    --------
    (child1, child2) : tuple[Chromosome, Chromosome]
    """
    # FIX: with only 1 block there is no valid cut point — clone instead
    if random.random() > crossover_rate or parent1.num_blocks < 2:
        return parent1.clone(), parent2.clone()

    n = parent1.num_blocks
    point = random.randint(1, n - 1)

    genes1 = parent1.genes[:point] + parent2.genes[point:]
    genes2 = parent2.genes[:point] + parent1.genes[point:]

    child1 = Chromosome(parent1.num_blocks, parent1.num_nodes,
                        parent1.replication, genes1)
    child2 = Chromosome(parent1.num_blocks, parent1.num_nodes,
                        parent1.replication, genes2)

    # Réparer les doublons éventuels
    child1.repair()
    child2.repair()

    return child1, child2


# ======================================================================
# Uniform Crossover
# ======================================================================

def uniform_crossover(parent1: Chromosome, parent2: Chromosome,
                      crossover_rate: float = 0.8,
                      mix_prob: float = 0.5):
    """
    Croisement uniforme : chaque gène est hérité de P1 ou P2 avec prob 0.5.

    Paramètres
    ----------
    parent1, parent2 : Chromosome
    crossover_rate   : float — Probabilité d'activer le croisement
    mix_prob         : float — Probabilité d'hériter du P1 pour chaque gène

    Retourne
    --------
    (child1, child2) : tuple[Chromosome, Chromosome]
    """
    if random.random() > crossover_rate:
        return parent1.clone(), parent2.clone()

    genes1, genes2 = [], []
    for g1, g2 in zip(parent1.genes, parent2.genes):
        if random.random() < mix_prob:
            genes1.append(list(g1))
            genes2.append(list(g2))
        else:
            genes1.append(list(g2))
            genes2.append(list(g1))

    child1 = Chromosome(parent1.num_blocks, parent1.num_nodes,
                        parent1.replication, genes1)
    child2 = Chromosome(parent1.num_blocks, parent1.num_nodes,
                        parent1.replication, genes2)

    child1.repair()
    child2.repair()

    return child1, child2