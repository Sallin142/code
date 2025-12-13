# CBS with Heuristics

## Introduction

This project implements and compares five different Conflict-Based Search (CBS) approaches.  CBS efficiently solves pathfinding problems for numerous agents by first planning individual paths and then resolving conflicts using constraint extension.  The implementation includes a standard CBS method, CBS disjoint splitting, and three heuristic variations that improve efficiency by employing various conflict resolution strategies: cardinal conflict detection, dependency graph, and weighted dependency graph.  These methods are intended to find collision-free, cost-effective pathways for several agents in grid-based settings.

This project implements **Conflict-Based Search (CBS)** for multi-agent pathfinding (MAPF) with five different algorithms:

- **CBS Standard** – basic CBS  
- **CBS Disjoint** – disjoint splitting  
- **CG** – cardinal conflict heuristic  
- **DG** – dependency graph heuristic  
- **WDG** – weighted dependency graph heuristic

### CBS Standard

CBS Standard is the baseline algorithm that resolves conflicts by splitting a high-level node into two children, each adding a negative constraint for one of the conflicting agents. The low-level planner then replans paths under these constraints using A*. This approach is simple and optimal but may explore many high-level nodes when conflicts are frequent.

### CBS Disjoint

CBS Disjoint improves upon standard CBS by using disjoint splitting, which introduces one positive and one negative constraint for each conflict. The positive constraint forces an agent to use a specific location or edge at a given timestep, reducing symmetry in the search space. This often leads to fewer high-level nodes compared to standard CBS while maintaining optimality.

### Cardinal Graph (CG) Heuristic

The CG heuristic focuses on cardinal conflicts, which are conflicts that cannot be avoided without increasing the cost of at least one agent. These conflicts are detected using Multi-valued Decision Diagrams (MDDs). A graph is constructed where vertices represent agents and edges represent cardinal conflicts. The heuristic value is computed as the size of a minimum vertex cover of this graph, providing a lower bound on the number of agents whose paths must increase in cost.

### Dependency Graph (DG) Heuristic

The DG heuristic generalizes the CG approach by identifying **dependencies** between agent pairs. Two agents are considered dependent if no pair of their individually optimal paths can be combined into a conflict-free joint solution. Dependencies are detected using joint MDD construction. A dependency graph is built with agents as vertices and dependencies as edges, and the heuristic value is again derived from a minimum vertex cover. DG captures more conflict structure than CG and therefore provides a stronger heuristic.

### Weighted Dependency Graph (WDG) Heuristic

The WDG heuristic further extends DG by assigning a weight to each dependency edge, representing the minimum additional cost required to resolve conflicts between the two agents. These weights are computed by solving a constrained two-agent planning problem. The heuristic is obtained by solving an edge-weighted minimum vertex cover problem, which estimates the total cost increase required to resolve all dependencies. Although more expensive to compute, WDG provides the strongest admissible heuristic among the implemented methods.

## Running the Benchmark

Run the benchmark script to compare all solvers:

```bash
python3 comprehensive_mapf_benchmark.py --repeat 1 --benchmark_dir custominstances/Benchmarks
```
