"""
mutation.py
Opérateurs de mutation pour l'algorithme génétique HDFS.

Deux variantes :
  1. Swap Mutation    — échange le nœud d'une réplique avec un nœud aléatoire
  2. Inversion Mutation — inverse un sous-segment du chromosome
"""

import random
from chromosome import Chromosome


# ======================================================================
# Swap Mutation (principale)
# ======================================================================

def swap_mutation(chrom: Chromosome, mutation_rate: float = 0.05) -> Chromosome:
    """
    Pour chaque réplique de chaque bloc, avec probabilité mutation_rate,
    remplace le nœud assigné par un autre nœud aléatoire distinct
    des autres répliques du même bloc.

    Exemple :
        Avant : B5 → [N1, N3, N4]   (mutation sur la 2e réplique)
        Après : B5 → [N1, N2, N4]   (N3 → N2)

    Paramètres
    ----------
    chrom         : Chromosome
    mutation_rate : float — Probabilité de mutation par gène

    Retourne
    --------
    Chromosome muté (in-place + return)
    """
    nodes = list(range(chrom.num_nodes))

    for i in range(chrom.num_blocks):
        for r_idx in range(chrom.replication):
            if random.random() < mutation_rate:
                current_replicas = list(chrom.genes[i])
                # FIX: removed two dead intermediate variable assignments;
                # keep only the correct final candidates list.
                # Candidates = all nodes except those already used by OTHER
                # replicas of this block (the current replica's slot is free).
                candidates = [
                    n for n in nodes
                    if n not in set(current_replicas) - {current_replicas[r_idx]}
                ]
                if candidates:
                    chrom.genes[i][r_idx] = random.choice(candidates)

    chrom.repair()   # Sécurité
    chrom.fitness = None
    return chrom


# ======================================================================
# Inversion Mutation (secondaire)
# ======================================================================

def inversion_mutation(chrom: Chromosome, mutation_rate: float = 0.05) -> Chromosome:
    """
    Avec probabilité mutation_rate, inverse un sous-segment aléatoire
    du chromosome (liste des blocs).

    Exemple (5 blocs) :
        Avant : [g0, g1, g2, g3, g4]
        Segment [1:4] inversé :
        Après : [g0, g3, g2, g1, g4]

    Paramètres
    ----------
    chrom         : Chromosome
    mutation_rate : float

    Retourne
    --------
    Chromosome muté
    """
    if random.random() < mutation_rate and chrom.num_blocks > 2:
        i = random.randint(0, chrom.num_blocks - 2)
        j = random.randint(i + 1, chrom.num_blocks - 1)
        chrom.genes[i:j+1] = chrom.genes[i:j+1][::-1]

    chrom.fitness = None
    return chrom