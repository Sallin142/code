# CBS with Heuristics

## Introduction

Multi-Agent Path Finding (MAPF) requires computing collision-free paths for multiple agents moving in a shared environment while minimizing the total cost. Conflict-Based Search (CBS) is a popular optimal MAPF algorithm that separates planning into a low-level path search and a high-level conflict resolution search. While CBS is optimal, its performance can degrade when many conflicts occur. To address this, several admissible heuristics have been proposed to solve the high-level search more effectively.

This project implements CBS together with multiple heuristic extensions that estimate the minimum additional cost required to resolve remaining conflicts. These heuristics improve search efficiency by reducing the number of expanded high-level nodes while preserving optimality. This project implements Conflict-Based Search (CBS) for multi-agent pathfinding (MAPF) with five different algorithms:

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

The CG heuristic focuses on **cardinal conflicts**, which are conflicts that cannot be avoided without increasing the cost of at least one agent. These conflicts are detected using Multi-valued Decision Diagrams (MDDs). A graph is constructed where vertices represent agents and edges represent cardinal conflicts. The heuristic value is computed as the size of a minimum vertex cover of this graph, providing a lower bound on the number of agents whose paths must increase in cost.

### Dependency Graph (DG) Heuristic

The DG heuristic generalizes the CG approach by identifying **dependencies** between agent pairs. Two agents are considered dependent if no pair of their individually optimal paths can be combined into a conflict-free joint solution. Dependencies are detected using joint MDD construction. A dependency graph is built with agents as vertices and dependencies as edges, and the heuristic value is again derived from a minimum vertex cover. DG captures more conflict structure than CG and therefore provides a stronger heuristic.

### Weighted Dependency Graph (WDG) Heuristic

The WDG heuristic further extends DG by assigning a weight to each dependency edge, representing the minimum additional cost required to resolve conflicts between the two agents. These weights are computed by solving a constrained two-agent planning problem. The heuristic is obtained by solving an edge-weighted minimum vertex cover problem, which estimates the total cost increase required to resolve all dependencies. Although more expensive to compute, WDG provides the strongest admissible heuristic among the implemented methods.

## Running the Benchmark

Run the benchmark script to compare all solvers:

```bash
python3 comprehensive_mapf_benchmark.py --repeat 1 --benchmark_dir custominstances/Benchmarks
```

To run particular algorithm on a specific MAPF instances:

``` bash
python3 run_experiments.py --instance custominstances/exp2_5.txt --solver WDGS
```
> Note: 
> - WDGS can be replaced by CBS, CGS, DGS
> - custominstances/exp2_5.txt can be replaced by any valid .txt MAPF instance file

## References

[1] Li, J., Felner, A., Boyarski, E., Ma, H., Koenig, S., University of Southern California, & Ben Gurion University of the Negev. (2019). Improved Heuristics for Multi-Agent Path Finding with Conflict-Based Search. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19). https://www.ijcai.org/proceedings/2019/0063.pdf

[2] Stern, R., Sturtevant, N., Felner, A., Koenig, S., Ma, H., Walker, T., Li, J., Atzmon, D., Cohen, L., Kumar, T. K. S., Boyarski, E., & Bartak, R. (2021). Multi-Agent Pathfinding: Definitions, variants, and benchmarks. Proceedings of the International Symposium on Combinatorial Search, 10(1), 151–158. https://movingai.com/benchmarks/mapf/index.html

