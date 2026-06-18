"""
compare.py
Comparaison : HDFS défaut vs HDFS Balancer vs PSO vs AG

Expérience 1 :
- Mesures réelles Hadoop : 5 DataNodes, réplication = 1, ~7 Go

Expérience 2 :
- Stress-test simulé complexe : 10 DataNodes, réplication = 3,
  3 racks simulés, capacités hétérogènes, 2000 blocs
"""

import random
import time
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithm'))

from fitness import FitnessEvaluator
from Genetic_algorithm import GeneticAlgorithm
from chromosome import Chromosome


NUM_NODES_REAL = 5
NODE_NAMES_REAL = ["DN:9866", "DN:9876", "DN:9886", "DN:9896", "DN:9906"]

LOADS_DEFAULT  = [1990, 866, 1008, 1240, 1830]
LOADS_BALANCER = [1280, 1496, 1008, 1388, 1966]

STRESS_NUM_BLOCKS   = 2000
STRESS_REPLICATION  = 3
STRESS_NUM_NODES    = 10

NODE_NAMES_STRESS = [f"DN:{9866 + i*10}" for i in range(STRESS_NUM_NODES)]
STRESS_BLOCK_SIZES = [128.0] * STRESS_NUM_BLOCKS

STRESS_NODE_CAPACITY = [
    150 * 1024,
     40 * 1024,
    200 * 1024,
     60 * 1024,
    180 * 1024,
     30 * 1024,
    160 * 1024,
     50 * 1024,
    190 * 1024,
     70 * 1024,
]

LOADS_STRESS = [
    180000,
      5000,
      3000,
    150000,
      4000,
      2000,
    140000,
      3500,
    278000,
      4500,
]

RACK_MAP = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]


def jain(loads):
    loads = np.array(loads, dtype=float)
    return (loads.sum() ** 2) / (len(loads) * np.sum(loads ** 2))


def capacity_overflow(loads, capacities):
    return sum(max(0, l - c) for l, c in zip(loads, capacities))


def print_metrics(label, loads_mo, node_names=None, capacities=None, extra=None):
    loads = np.array(loads_mo, dtype=float)
    names = node_names or [f"DN{i}" for i in range(len(loads))]

    print(f"\n{'═' * 70}")
    print(f"  {label}")
    print(f"{'═' * 70}")

    scale = max(1000, loads.max() / 40)

    for i, (name, value) in enumerate(zip(names, loads.tolist())):
        bar = '' * int(value / scale)

        if capacities:
            cap = capacities[i]
            overflow = max(0, value - cap)
            status = f" | cap={cap:.0f} Mo"
            if overflow > 0:
                status += f" | OVERFLOW={overflow:.0f} Mo ❌"
            else:
                status += " | OK ✅"
        else:
            status = ""

        print(f"  {name} : {value:10.1f} Mo  {bar}{status}")

    print(f"  {'─' * 65}")
    print(f"  Moyenne        : {np.mean(loads):10.2f} Mo")
    print(f"  Écart-type     : {np.std(loads):10.4f} Mo")
    print(f"  Indice de Jain : {jain(loads):10.4f}  (1.0 = parfait)")

    if capacities:
        overflow_total = capacity_overflow(loads, capacities)
        print(f"  Dépassement capacité total : {overflow_total:.1f} Mo")

    if extra:
        for k, v in extra.items():
            print(f"  {k:<22}: {v}")


class Particle:
    def __init__(self, num_blocks, num_nodes, replication):
        nodes = list(range(num_nodes))
        self.position = [
            random.sample(nodes, replication)
            for _ in range(num_blocks)
        ]
        self.best_position = [list(p) for p in self.position]
        self.best_fitness  = -float("inf")

    def update(self, global_best, num_blocks, w=0.5, c1=1.5, c2=1.5):
        new_pos = []
        for i in range(num_blocks):
            r1, r2 = random.random(), random.random()
            if r2 * c2 > r1 * c1 and r2 * c2 > w:
                new_pos.append(list(global_best[i]))
            elif r1 * c1 > w:
                new_pos.append(list(self.best_position[i]))
            else:
                new_pos.append(list(self.position[i]))
        self.position = new_pos


def run_pso(evaluator, num_blocks, num_nodes, replication,
            n_particles=50, max_iter=200, patience=30, seed=42):

    random.seed(seed)
    swarm = [Particle(num_blocks, num_nodes, replication)
             for _ in range(n_particles)]

    global_best_pos = None
    global_best_fit = -float("inf")
    no_improve      = 0
    history         = []

    t_start = time.time()

    for it in range(max_iter):
        improved = False

        for p in swarm:
            chrom = Chromosome(num_blocks, num_nodes, replication,
                               genes=p.position)
            fit = evaluator.evaluate(chrom)

            if fit > p.best_fitness:
                p.best_fitness  = fit
                p.best_position = [list(x) for x in p.position]

            if fit > global_best_fit:
                global_best_fit = fit
                global_best_pos = [list(x) for x in p.position]
                improved        = True

        history.append(global_best_fit)
        no_improve = 0 if improved else no_improve + 1

        if no_improve >= patience:
            print(f"  [PSO] Convergence à l'itération {it}")
            break

        for p in swarm:
            p.update(global_best_pos, num_blocks)

    pso_time = round(time.time() - t_start, 3)
    return global_best_pos, global_best_fit, history, len(history), pso_time


print("\n" + "═" * 70)
print("  EXPÉRIENCE 1 — MESURES RÉELLES HADOOP")
print("═" * 70)

print_metrics("1. HDFS par défaut mesuré", LOADS_DEFAULT, node_names=NODE_NAMES_REAL)
print_metrics("2. HDFS Balancer mesuré", LOADS_BALANCER, node_names=NODE_NAMES_REAL)


print("\n" + "═" * 70)
print("  EXPÉRIENCE 2 — STRESS-TEST COMPLEXE")
print("═" * 70)

print_metrics(
    "3. Stress-test initial",
    LOADS_STRESS,
    node_names=NODE_NAMES_STRESS,
    capacities=STRESS_NODE_CAPACITY
)

stress_evaluator = FitnessEvaluator(
    block_sizes=STRESS_BLOCK_SIZES,
    node_capacity=STRESS_NODE_CAPACITY,
    node_load=LOADS_STRESS,
    rack_map=RACK_MAP,
    w1=0.5,
    w2=0.3,
    w3=0.2,
)

print("\n   Lancement PSO...")
pso_pos, pso_fit, pso_history, pso_iters, pso_time = run_pso(
    evaluator=stress_evaluator,
    num_blocks=STRESS_NUM_BLOCKS,
    num_nodes=STRESS_NUM_NODES,
    replication=STRESS_REPLICATION,
    n_particles=50,
    max_iter=200,
    patience=30,
    seed=42,
)

pso_chrom = Chromosome(
    STRESS_NUM_BLOCKS,
    STRESS_NUM_NODES,
    STRESS_REPLICATION,
    genes=pso_pos
)

stress_evaluator.evaluate(pso_chrom)
pso_bd = stress_evaluator.breakdown(pso_chrom)
pso_overflow = capacity_overflow(pso_bd["NodeLoads_MB"], STRESS_NODE_CAPACITY)

print_metrics(
    "4. PSO — après optimisation",
    pso_bd["NodeLoads_MB"],
    node_names=NODE_NAMES_STRESS,
    capacities=STRESS_NODE_CAPACITY,
    extra={
        "Fitness": round(pso_bd["Fitness"], 4),
        "Localité (rack)": round(pso_bd["Locality"], 4),
        "Migration": round(pso_bd["MigrationCost"], 4),
        "CapacityPenalty": round(pso_bd["CapacityPenalty"], 4),
        "Overflow total": round(pso_overflow, 2),
        "Itérations": pso_iters,
        "Temps (s)": pso_time,
    }
)

print("\n  Lancement AG...")

ga = GeneticAlgorithm(
    evaluator=stress_evaluator,
    num_blocks=STRESS_NUM_BLOCKS,
    num_nodes=STRESS_NUM_NODES,
    replication=STRESS_REPLICATION,
    pop_size=100,
    max_generations=300,
    crossover_rate=0.85,
    mutation_rate=0.03,
    elite_size=5,
    tournament_k=4,
    patience=50,
    seed=42,
)

best  = ga.run(verbose=False)
ag_bd = stress_evaluator.breakdown(best)
ag_overflow = capacity_overflow(ag_bd["NodeLoads_MB"], STRESS_NODE_CAPACITY)

print_metrics(
    "5. AG — après optimisation",
    ag_bd["NodeLoads_MB"],
    node_names=NODE_NAMES_STRESS,
    capacities=STRESS_NODE_CAPACITY,
    extra={
        "Fitness": round(ag_bd["Fitness"], 4),
        "Localité (rack)": round(ag_bd["Locality"], 4),
        "Migration": round(ag_bd["MigrationCost"], 4),
        "CapacityPenalty": round(ag_bd["CapacityPenalty"], 4),
        "Overflow total": round(ag_overflow, 2),
        "Générations": ga.num_generations_run,
        "Temps (s)": round(ga.execution_time, 3),
    }
)


print("\n\n" + "═" * 100)
print(f"{'TABLEAU COMPARATIF FINAL':^100}")
print("═" * 100)
print(f"  {'Méthode':<38} {'Std(Mo)':>12} {'Jain':>10} {'Overflow(Mo)':>15} {'Fitness':>15} {'Temps':>10}")
print("  " + "─" * 92)

results = [
    ("HDFS par défaut mesuré", LOADS_DEFAULT, 0, "-", "-", None),
    ("HDFS Balancer mesuré", LOADS_BALANCER, 0, "-", "-", None),
    ("Stress-test initial", LOADS_STRESS, capacity_overflow(LOADS_STRESS, STRESS_NODE_CAPACITY), "-", "-", STRESS_NODE_CAPACITY),
    ("PSO stress-test", pso_bd["NodeLoads_MB"], pso_overflow, round(pso_bd["Fitness"], 4), f"{pso_time}s", STRESS_NODE_CAPACITY),
    ("AG stress-test", ag_bd["NodeLoads_MB"], ag_overflow, round(ag_bd["Fitness"], 4), f"{round(ga.execution_time,3)}s", STRESS_NODE_CAPACITY),
]

for name, loads, overflow, fit, t, capacities in results:
    arr = np.array(loads, dtype=float)
    flag = " ✅" if name == "AG stress-test" else ""
    print(
        f"  {name:<38} "
        f"{np.std(arr):>12.4f} "
        f"{jain(arr):>10.4f} "
        f"{overflow:>15} "
        f"{str(fit):>15} "
        f"{str(t):>10}{flag}"
    )

