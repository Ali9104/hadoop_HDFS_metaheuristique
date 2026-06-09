"""
chromosome.py
Représentation d'un chromosome pour le placement de blocs HDFS.
Chaque chromosome encode une solution complète de placement.
"""

import random
import copy


class Chromosome:
    """
    Un chromosome représente une solution de placement de blocs HDFS.

    Structure :
        genes : liste de listes
            genes[i] = liste des nœuds hébergeant les R répliques du bloc i
            Exemple avec 4 blocs, R=3, 5 nœuds :
            [[N1, N3, N2], [N2, N4, N1], [N3, N1, N5], [N2, N3, N4]]

    Attributs
    ---------
    genes       : list[list[int]]   — Affectation bloc → nœuds
    num_blocks  : int               — Nombre de blocs (n)
    num_nodes   : int               — Nombre de nœuds (m)
    replication : int               — Facteur de réplication (R, défaut 3)
    fitness     : float             — Valeur de fitness (calculée séparément)
    """

    def __init__(self, num_blocks: int, num_nodes: int, replication: int = 3,
                 genes=None):
        # FIX: guard against impossible configuration
        if replication > num_nodes:
            raise ValueError(
                f"replication ({replication}) cannot exceed num_nodes ({num_nodes})"
            )
        self.num_blocks = num_blocks
        self.num_nodes = num_nodes
        self.replication = replication
        self.fitness = None

        if genes is not None:
            self.genes = copy.deepcopy(genes)
        else:
            self.genes = self._random_init()

    # ------------------------------------------------------------------
    # Initialisation aléatoire
    # ------------------------------------------------------------------
    def _random_init(self) -> list:
        """
        Génère un chromosome aléatoire valide.
        Pour chaque bloc, choisit R nœuds distincts (contrainte HDFS).
        """
        genes = []
        nodes = list(range(self.num_nodes))
        for _ in range(self.num_blocks):
            replicas = random.sample(nodes, self.replication)
            genes.append(replicas)
        return genes

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def is_valid(self) -> bool:
        """Vérifie que chaque bloc a exactement R répliques distinctes."""
        for block_replicas in self.genes:
            if len(block_replicas) != self.replication:
                return False
            if len(set(block_replicas)) != self.replication:
                return False
        return True

    def repair(self):
        """
        Répare un chromosome invalide (doublons après croisement/mutation).
        FIX: was an infinite loop when num_nodes < replication. Now raises
        a clear error instead of hanging forever.
        """
        if self.replication > self.num_nodes:
            raise ValueError(
                f"Cannot repair: replication ({self.replication}) > "
                f"num_nodes ({self.num_nodes})"
            )
        nodes = list(range(self.num_nodes))
        for i, block_replicas in enumerate(self.genes):
            unique = list(dict.fromkeys(block_replicas))  # dédoublonnage ordonné
            if len(unique) < self.replication:
                available = [n for n in nodes if n not in unique]
                random.shuffle(available)
                unique.extend(available[:self.replication - len(unique)])
            self.genes[i] = unique[:self.replication]

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------
    def clone(self):
        c = Chromosome(self.num_blocks, self.num_nodes,
                       self.replication, self.genes)
        c.fitness = self.fitness
        return c

    def __repr__(self):
        lines = [f"Chromosome (fitness={self.fitness:.4f})" if self.fitness is not None
                 else "Chromosome (fitness=None)"]
        for i, replicas in enumerate(self.genes):
            node_names = [f"N{r+1}" for r in replicas]
            lines.append(f"  B{i+1} → {node_names}")
        return "\n".join(lines)