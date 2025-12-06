# CBS with Heuristics

This project implements **Conflict-Based Search (CBS)** for multi-agent pathfinding (MAPF) with five different algorithms:

- **CBS Standard** – basic CBS  
- **CBS Disjoint** – disjoint splitting  
- **CG** – cardinal conflict heuristic  
- **DG** – dependency graph heuristic  
- **WDG** – weighted dependency graph heuristic

## Running the Benchmark

Run the benchmark script to compare all solvers:

```bash
python3 comprehensive_mapf_benchmark.py --repeat 1 --benchmark_dir custominstances/Benchmarks
