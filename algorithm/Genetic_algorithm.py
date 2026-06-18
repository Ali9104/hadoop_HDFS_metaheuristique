"""
genetic_algorithm.py
Algorithme Génétique complet pour le placement optimisé de blocs HDFS.

Boucle principale :
    Initialisation
    ↓  Évaluation Fitness
    ↓  Sélection
    ↓  Croisement
    ↓  Mutation
    ↓  Élitisme + Nouvelle génération
    ↓  Convergence → Solution optimale
"""

import random
import time
import math
from chromosome  import Chromosome
from fitness     import FitnessEvaluator
from selection   import tournament_selection, elitism_selection
from crossover   import single_point_crossover
from mutation    import swap_mutation


# ======================================================================
# Algorithme Génétique
# ======================================================================

class GeneticAlgorithm:
    """
    AG pour l'optimisation du placement de blocs HDFS.

    Paramètres
    ----------
    evaluator       : FitnessEvaluator
    num_blocks      : int   — n (nombre de blocs)
    num_nodes       : int   — m (nombre de nœuds)
    replication     : int   — R (facteur de réplication, défaut 3)
    pop_size        : int   — Taille de la population
    max_generations : int   — Nombre maximum de générations
    crossover_rate  : float — Probabilité de croisement
    mutation_rate   : float — Probabilité de mutation par gène
    elite_size      : int   — Nombre d'individus élites conservés
    tournament_k    : int   — Taille du tournoi
    patience        : int   — Convergence : arrêt si pas d'amélioration après N générations
    """

    def __init__(self,
                 evaluator: FitnessEvaluator,
                 num_blocks: int,
                 num_nodes: int,
                 replication: int = 3,
                 pop_size: int = 50,
                 max_generations: int = 200,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.05,
                 elite_size: int = 5,
                 tournament_k: int = 3,
                 patience: int = 30,
                 seed: int = None):

        # FIX: validate elite_size vs pop_size to prevent empty offspring list
        if elite_size >= pop_size:
            raise ValueError(
                f"elite_size ({elite_size}) must be less than pop_size ({pop_size})"
            )

        self.evaluator       = evaluator
        self.num_blocks      = num_blocks
        self.num_nodes       = num_nodes
        self.replication     = replication
        self.pop_size        = pop_size
        self.max_generations = max_generations
        self.crossover_rate  = crossover_rate
        self.mutation_rate   = mutation_rate
        self.elite_size      = elite_size
        self.tournament_k    = tournament_k
        self.patience        = patience

        if seed is not None:
            random.seed(seed)

        # Historique pour analyse
        self.history = {
            "best_fitness":  [],
            "avg_fitness":   [],
            "worst_fitness": [],
            "generation":    [],
        }

        self.best_solution  = None
        self.best_fitness   = -math.inf
        self.num_generations_run = 0
        self.execution_time = 0.0

    # ------------------------------------------------------------------
    # Initialisation de la population
    # ------------------------------------------------------------------
    def _init_population(self) -> list:
        """Génère pop_size chromosomes aléatoires valides."""
        population = []
        for _ in range(self.pop_size):
            chrom = Chromosome(self.num_blocks, self.num_nodes, self.replication)
            population.append(chrom)
        return population

    # ------------------------------------------------------------------
    # Évaluation
    # ------------------------------------------------------------------
    def _evaluate_population(self, population: list):
        """Calcule la fitness de tous les individus non évalués."""
        for chrom in population:
            if chrom.fitness is None:
                self.evaluator.evaluate(chrom)

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------
    def run(self, verbose: bool = True) -> Chromosome:
        """
        Lance l'algorithme génétique.
        Retourne le meilleur chromosome trouvé.
        """
        start_time = time.time()

        # ── Étape 1 : Initialisation ──────────────────────────────────
        population = self._init_population()
        self._evaluate_population(population)

        no_improve_counter = 0

        for gen in range(self.max_generations):

            # ── Étape 2 : Statistiques ────────────────────────────────
            fits = [c.fitness for c in population]
            best_fit_gen  = max(fits)
            avg_fit_gen   = sum(fits) / len(fits)
            worst_fit_gen = min(fits)

            self.history["generation"].append(gen)
            self.history["best_fitness"].append(best_fit_gen)
            self.history["avg_fitness"].append(avg_fit_gen)
            self.history["worst_fitness"].append(worst_fit_gen)

            # Mettre à jour le meilleur global
            if best_fit_gen > self.best_fitness:
                best_idx = fits.index(best_fit_gen)
                self.best_solution = population[best_idx].clone()
                self.best_fitness  = best_fit_gen
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if verbose and gen % 10 == 0:
                print(f"Gén {gen:4d} | Best={best_fit_gen:.4f} "
                      f"| Avg={avg_fit_gen:.4f} "
                      f"| Worst={worst_fit_gen:.4f}")

            # ── Critère d'arrêt anticipé ──────────────────────────────
            if no_improve_counter >= self.patience:
                if verbose:
                    print(f"\n[Convergence] Arrêt à la génération {gen} "
                          f"(pas d'amélioration depuis {self.patience} générations)")
                break

            # ── Étape 3 : Élitisme ────────────────────────────────────
            elites = elitism_selection(population, self.elite_size)

            # ── Étape 4 : Sélection ───────────────────────────────────
            num_parents = self.pop_size - self.elite_size
            parents = tournament_selection(population, num_parents,
                                           self.tournament_k)

            # ── Étape 5 : Croisement ──────────────────────────────────
            offspring = []
            random.shuffle(parents)
            for i in range(0, len(parents) - 1, 2):
                c1, c2 = single_point_crossover(parents[i], parents[i+1],
                                                self.crossover_rate)
                offspring.extend([c1, c2])
            if len(parents) % 2 == 1:
                offspring.append(parents[-1].clone())

            # Ajuster à la bonne taille
            offspring = offspring[:num_parents]

            # ── Étape 6 : Mutation ────────────────────────────────────
            for chrom in offspring:
                swap_mutation(chrom, self.mutation_rate)

            # ── Étape 7 : Nouvelle génération ─────────────────────────
            population = elites + offspring
            self._evaluate_population(population)

            self.num_generations_run = gen + 1

        self.execution_time = time.time() - start_time

        if verbose:
            self._print_results()

        return self.best_solution

    # ------------------------------------------------------------------
    # Affichage des résultats
    # ------------------------------------------------------------------
    def _print_results(self):
        print("\n" + "="*60)
        print("RÉSULTATS DE L'ALGORITHME GÉNÉTIQUE")
        print("="*60)
        print(f"Générations exécutées : {self.num_generations_run}")
        print(f"Temps d'exécution     : {self.execution_time:.3f} s")
        print(f"Meilleure fitness     : {self.best_fitness:.4f}")
        print("\nDétail de la solution optimale :")
        breakdown = self.evaluator.breakdown(self.best_solution)
        for k, v in breakdown.items():
            print(f"  {k}: {v}")
        print("\nPlacement optimal :")
        print(self.best_solution)

    def get_results(self) -> dict:
        """Retourne un dictionnaire structuré pour comparaison / rapport."""
        return {
            "best_fitness":       self.best_fitness,
            "num_generations":    self.num_generations_run,
            "execution_time_s":   round(self.execution_time, 4),
            "best_solution":      self.best_solution,
            "fitness_history":    self.history,
            "breakdown":          self.evaluator.breakdown(self.best_solution),
        }


# ======================================================================
# Script de démonstration
# ======================================================================

if __name__ == "__main__":
    import random

    # ── Configuration du cluster ──────────────────────────────────────
    NUM_BLOCKS = 20
    NUM_NODES  = 5
    REPLICATION = 3

    # Tailles des blocs en Mo (simulation : entre 64 Mo et 256 Mo)
    random.seed(42)
    block_sizes = [random.uniform(64, 256) for _ in range(NUM_BLOCKS)]

    # Capacités des nœuds en Mo
    node_capacity = [4096, 4096, 8192, 4096, 8192]

    # Charge initiale (données déjà présentes)
    node_load_init = [512, 256, 1024, 768, 512]

    # Mapping nœud → rack  (5 nœuds, 3 racks)
    rack_map = [0, 0, 1, 1, 2]

    # ── Évaluateur de fitness ─────────────────────────────────────────
    evaluator = FitnessEvaluator(
        block_sizes=block_sizes,
        node_capacity=node_capacity,
        node_load=node_load_init,
        rack_map=rack_map,
        w1=0.5, w2=0.3, w3=0.2,
    )

    # ── Algorithme génétique ──────────────────────────────────────────
    ga = GeneticAlgorithm(
        evaluator=evaluator,
        num_blocks=NUM_BLOCKS,
        num_nodes=NUM_NODES,
        replication=REPLICATION,
        pop_size=100,
        max_generations=300,
        crossover_rate=0.85,
        mutation_rate=0.03,
        elite_size=5,
        tournament_k=4,
        patience=50,
        seed=42,
    )

    best = ga.run(verbose=True)
    results = ga.get_results()

    print("\n[Résultats pour comparaison]")
    print(f"  Fitness       : {results['best_fitness']:.4f}")
    print(f"  Générations   : {results['num_generations']}")
    print(f"  Temps (s)     : {results['execution_time_s']}")