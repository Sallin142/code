#!/usr/bin/env python3

import sys
from CGSolver import CGSolver
from DGSolver import DGSolver
from WDGSolver import WDGSolver
from cbs import CBSSolver

def import_mapf_instance(filename):
    with open(filename, 'r') as f:

        line = f.readline()
        rows, cols = [int(x) for x in line.split()]
        
        my_map = []
        for r in range(rows):
            line = f.readline()
            row = []
            for cell in line.split():
                row.append(cell == '@')
            my_map.append(row)
        
        num_agents = int(f.readline())
        starts = []
        goals = []
        for a in range(num_agents):
            sx, sy, gx, gy = [int(x) for x in f.readline().split()]
            starts.append((sx, sy))
            goals.append((gx, gy))
    
    return my_map, starts, goals

def test_solver(solver_name, my_map, starts, goals):
    print(f"\n{'='*60}")
    print(f"Testing {solver_name}")
    print(f"{'='*60}")
    
    try:
        if solver_name == "CG":
            solver = CGSolver(my_map, starts, goals)
        elif solver_name == "DG":
            solver = DGSolver(my_map, starts, goals)
        elif solver_name == "WDG":
            solver = WDGSolver(my_map, starts, goals)
        elif solver_name == "CBS":
            solver = CBSSolver(my_map, starts, goals)
        else:
            print(f"Unknown solver: {solver_name}")
            return None
        
        paths = solver.find_solution(disjoint=True)
        
        if paths:
            print(f"✓ SUCCESS!")
            print(f"  Expanded nodes:  {solver.num_of_expanded}")
            print(f"  Generated nodes: {solver.num_of_generated}")
            print(f"  CPU time:        {solver.CPU_time:.3f}s")
            
            # Get root h-value if available
            root_h = None
            if hasattr(solver, 'root_h_value'):
                root_h = solver.root_h_value
                print(f"  Root h-value:    {root_h}")
            
            return {
                'expanded': solver.num_of_expanded,
                'generated': solver.num_of_generated,
                'time': solver.CPU_time,
                'root_h': root_h
            }
        else:
            print(f"x FAILED - No solution found")
            return None
            
    except Exception as e:
        print(f"x ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    
    filename = sys.argv[1]
    
    print(f"\n{'='*60}")
    print(f"Loading instance: {filename}")
    print(f"{'='*60}")
    
    try:
        my_map, starts, goals = import_mapf_instance(filename)
        print(f"Map size: {len(my_map)}x{len(my_map[0])}")
        print(f"Agents: {len(starts)}")
        
        results = {}
        for solver_name in ['CBS', 'CG', 'DG', 'WDG']:
            result = test_solver(solver_name, my_map, starts, goals)
            if result:
                results[solver_name] = result
    
        if results:
            print(f"\n{'='*60}")
            print("COMPARISON")
            print(f"{'='*60}")
            
            # Print header
            header_parts = ['Solver', 'Expanded', 'Generated', 'Time (s)']
            if any(r.get('root_h') is not None for r in results.values()):
                header_parts.append('Root h')
            
            print(f"{header_parts[0]:<10} {header_parts[1]:<12} {header_parts[2]:<12} {header_parts[3]:<10}", end='')
            if len(header_parts) > 4:
                print(f" {header_parts[4]:<10}", end='')
            print()
            print("-" * (60 + (12 if len(header_parts) > 4 else 0)))
            
            # Print results
            for solver_name in ['CBS', 'CG', 'DG', 'WDG']:
                if solver_name in results:
                    r = results[solver_name]
                    print(f"{solver_name:<10} {r['expanded']:<12} {r['generated']:<12} {r['time']:<10.3f}", end='')
                    if r.get('root_h') is not None:
                        print(f" {r['root_h']:<10}", end='')
                    elif any(res.get('root_h') is not None for res in results.values()):
                        print(f" {'N/A':<10}", end='')
                    print()

            # Calculate improvements
            print(f"\n{'='*60}")
            print("IMPROVEMENTS")
            print(f"{'='*60}")
            
            if 'CBS' in results and 'CG' in results:
                node_reduction = (results['CBS']['expanded'] - results['CG']['expanded']) / results['CBS']['expanded'] * 100
                time_speedup = results['CBS']['time'] / results['CG']['time'] if results['CG']['time'] > 0 else float('inf')
                print(f"CG vs CBS:")
                print(f"  Node reduction: {node_reduction:.1f}%")
                print(f"  Speedup: {time_speedup:.2f}x")
            
            if 'CG' in results and 'DG' in results:
                node_reduction = (results['CG']['expanded'] - results['DG']['expanded']) / results['CG']['expanded'] * 100
                time_speedup = results['CG']['time'] / results['DG']['time'] if results['DG']['time'] > 0 else float('inf')
                print(f"\nDG vs CG:")
                print(f"  Node reduction: {node_reduction:.1f}%")
                print(f"  Speedup: {time_speedup:.2f}x")
            
            if 'DG' in results and 'WDG' in results:
                node_reduction = (results['DG']['expanded'] - results['WDG']['expanded']) / results['DG']['expanded'] * 100
                time_speedup = results['DG']['time'] / results['WDG']['time'] if results['WDG']['time'] > 0 else float('inf')
                print(f"\nWDG vs DG:")
                print(f"  Node reduction: {node_reduction:.1f}%")
                print(f"  Speedup: {time_speedup:.2f}x")
            
            if 'CBS' in results and 'WDG' in results:
                node_reduction = (results['CBS']['expanded'] - results['WDG']['expanded']) / results['CBS']['expanded'] * 100
                time_speedup = results['CBS']['time'] / results['WDG']['time'] if results['WDG']['time'] > 0 else float('inf')
                print(f"\nWDG vs CBS (Overall):")
                print(f"  Node reduction: {node_reduction:.1f}%")
                print(f"  Speedup: {time_speedup:.2f}x")
                
    except FileNotFoundError:
        print(f"Error: can't read the file: {filename}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()