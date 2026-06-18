"""
fitness.py
Fonction de fitness multi-critère pour le placement de blocs HDFS.

Fitness = w1 * LoadBalance + w2 * Locality - w3 * MigrationCost

Références mathématiques
-------------------------
Variables :
    B = {b_1, ..., b_n}   ensemble des blocs
    N = {n_1, ..., n_m}   ensemble des nœuds (DataNodes)
    x_ij ∈ {0,1}          1 si le bloc i est placé sur le nœud j

Contraintes :
    ∑_j x_ij = R          pour tout bloc i  (réplication exacte)
    ∑_i size_i * x_ij ≤ Capacity_j          (capacité nœud)

Objectif principal :
    min ∑_j (Load_j - Load̄)²
    où  Load_j = ∑_i size_i * x_ij
        Load̄   = (∑_j Load_j) / m
"""

import math
from chromosome import Chromosome


# ======================================================================
# Classe principale
# ======================================================================

class FitnessEvaluator:
    """
    Évalue la qualité d'une solution de placement.

    Paramètres
    ----------
    block_sizes   : list[float]  — Tailles des blocs en Mo (longueur n)
    node_capacity : list[float]  — Capacités des nœuds en Mo (longueur m)
    node_load     : list[float]  — Charge actuelle des nœuds (avant migration)
    rack_map      : list[int]    — rack_map[j] = ID du rack du nœud j
    w1, w2, w3    : float        — Poids des trois composantes
    penalty       : float        — Pénalité si contrainte capacité violée
    """

    def __init__(self,
                 block_sizes: list,
                 node_capacity: list,
                 node_load: list = None,
                 rack_map: list = None,
                 w1: float = 0.5,
                 w2: float = 0.3,
                 w3: float = 0.2,
                 penalty: float = 1e6):
        self.block_sizes   = block_sizes
        self.node_capacity = node_capacity
        self.node_load_init = node_load if node_load else [0.0] * len(node_capacity)
        self.rack_map = rack_map if rack_map else list(range(len(node_capacity)))
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.penalty = penalty
        self.num_nodes  = len(node_capacity)
        self.num_blocks = len(block_sizes)

        # FIX: validate that lengths are consistent to prevent silent IndexError
        if len(self.node_load_init) != self.num_nodes:
            raise ValueError(
                f"node_load length ({len(self.node_load_init)}) must equal "
                f"num_nodes ({self.num_nodes})"
            )
        if len(self.rack_map) != self.num_nodes:
            raise ValueError(
                f"rack_map length ({len(self.rack_map)}) must equal "
                f"num_nodes ({self.num_nodes})"
            )

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------
    def evaluate(self, chrom: Chromosome) -> float:
        """
        Calcule et stocke chrom.fitness.
        Retourne la valeur scalaire.

        FIX: added guard so a chromosome with more nodes than the evaluator
        knows about raises a clear error instead of a silent IndexError.
        """
        if chrom.num_nodes != self.num_nodes:
            raise ValueError(
                f"Chromosome num_nodes ({chrom.num_nodes}) does not match "
                f"evaluator num_nodes ({self.num_nodes})"
            )

        # 1. Vérifier contrainte de capacité
        loads = self._compute_loads(chrom)
        penalty = self._capacity_penalty(loads)

        # 2. Composante LoadBalance (à maximiser → on l'inverse du déséquilibre)
        lb = self._load_balance(loads)

        # 3. Composante Locality (à maximiser)
        loc = self._locality(chrom)

        # 4. Composante MigrationCost (à minimiser → soustraction)
        migr = self._migration_cost(chrom, loads)

        fitness = self.w1 * lb + self.w2 * loc - self.w3 * migr - penalty
        chrom.fitness = fitness
        return fitness

    # ------------------------------------------------------------------
    # Composante 1 : Load Balance
    # ------------------------------------------------------------------
    def _compute_loads(self, chrom: Chromosome) -> list:
        """
        Load_j = ∑_i size_i * x_ij
        Retourne une liste de charges par nœud.
        """
        loads = [0.0] * self.num_nodes
        for i, replicas in enumerate(chrom.genes):
            for j in replicas:
                loads[j] += self.block_sizes[i]
        return loads

    def _load_balance(self, loads: list) -> float:
        """
        Mesure d'équilibre.
        LoadBalance = 1 - (σ / μ)   où σ = écart-type, μ = moyenne
        Valeur entre 0 et 1 (1 = parfaitement équilibré).
        """
        mean_load = sum(loads) / self.num_nodes if self.num_nodes > 0 else 0
        if mean_load == 0:
            return 1.0
        variance = sum((l - mean_load) ** 2 for l in loads) / self.num_nodes
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_load          # Coefficient de variation
        balance = max(0.0, 1.0 - cv)     # ∈ [0, 1]
        return balance

    # ------------------------------------------------------------------
    # Composante 2 : Locality (conformité rack HDFS)
    # ------------------------------------------------------------------
    def _locality(self, chrom: Chromosome) -> float:
        """
        Locality score basé sur la politique de placement HDFS :
          - Réplique 1 sur un rack
          - Réplique 2 sur un rack différent
          - Réplique 3 sur un nœud différent du même rack que la 2e
        Score = fraction de blocs respectant la diversité de rack.
        1.0 → toutes les répliques sur des racks différents.
        """
        if self.num_blocks == 0:
            return 1.0
        score = 0.0
        for replicas in chrom.genes:
            racks = [self.rack_map[j] for j in replicas]
            distinct_racks = len(set(racks))
            # Normaliser : 1 seul rack = 0, tous distincts = 1
            ratio = (distinct_racks - 1) / (len(replicas) - 1) if len(replicas) > 1 else 1.0
            score += ratio
        return score / self.num_blocks

    # ------------------------------------------------------------------
    # Composante 3 : Migration Cost
    # ------------------------------------------------------------------
    def _migration_cost(self, chrom: Chromosome, loads: list) -> float:
        """
        Coût normalisé de migration.
        MigrationCost = ∑_i size_i  pour les blocs dont la répartition
        diffère de l'affectation initiale (si fournie via node_load_init).
        Normalisé par la taille totale des blocs.
        """
        total_size = sum(self.block_sizes)
        if total_size == 0:
            return 0.0
        # Approximation : la charge dépassant la charge initiale
        #   représente les données migrées
        migrated = 0.0
        for j in range(self.num_nodes):
            excess = loads[j] - self.node_load_init[j]
            if excess > 0:
                migrated += excess
        return migrated / total_size     # ∈ [0, ∞) normalisé

    # ------------------------------------------------------------------
    # Pénalité capacité
    # ------------------------------------------------------------------
    def _capacity_penalty(self, loads: list) -> float:
        """
        Pénalité appliquée si ∑_i size_i * x_ij > Capacity_j.
        Pénalité proportionnelle au dépassement.
        """
        pen = 0.0
        for j, load in enumerate(loads):
            overflow = load - self.node_capacity[j]
            if overflow > 0:
                pen += self.penalty * overflow
        return pen

    # ------------------------------------------------------------------
    # Décomposition pour rapport / debug
    # ------------------------------------------------------------------
    def breakdown(self, chrom: Chromosome) -> dict:
        """Retourne les composantes individuelles pour analyse."""
        loads = self._compute_loads(chrom)
        pen   = self._capacity_penalty(loads)
        lb    = self._load_balance(loads)
        loc   = self._locality(chrom)
        migr  = self._migration_cost(chrom, loads)
        total = self.w1 * lb + self.w2 * loc - self.w3 * migr - pen
        return {
            "LoadBalance":     round(lb,   4),
            "Locality":        round(loc,  4),
            "MigrationCost":   round(migr, 4),
            "CapacityPenalty": round(pen,  4),
            "Fitness":         round(total,4),
            "NodeLoads_MB":    [round(l, 2) for l in loads],
        }