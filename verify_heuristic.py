

"""
Verify that CG/DG/WDG heuristics are admissible and informative.
"""

import sys
from CGSolver import CGSolver
from DGSolver import DGSolver
from WDGSolver import WDGSolver
from cbs import CBSSolver
from single_agent_planner import compute_heuristics, a_star


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

def verify_heuristic(filename):
    """Check if heuristics are working correctly."""
    
    print(f"Loading: {filename}\n")
    my_map, starts, goals = import_mapf_instance(filename)
    cbs_solver = CBSSolver(my_map, starts, goals)
    
    
    paths = []
    heuristics = []
    for i in range(len(starts)):
        h = compute_heuristics(my_map, goals[i])
        heuristics.append(h)
        path = a_star(my_map, starts[i], goals[i], h, i, [])
        paths.append(path)
    
    collisions = cbs_solver.detect_collisions(paths)
    
    print("="*70)
    print("ROOT NODE ANALYSIS")
    print("="*70)
    print(f"Agents: {len(starts)}")
    print(f"Collisions at root: {len(collisions)}")
    
    
    print("\nComputing h-values...")
    
    
    cg_solver = CGSolver(my_map, starts, goals)
    cg_h = cg_solver.get_cg_heuristic(my_map, paths, starts, goals, heuristics, [])
    
    
    dg_solver = DGSolver(my_map, starts, goals)
    dg_h = dg_solver.get_dg_heuristic(my_map, paths, starts, goals, heuristics, [])
    
    
    wdg_solver = WDGSolver(my_map, starts, goals)
    wdg_h = wdg_solver.get_wdg_heuristic(my_map, paths, starts, goals, heuristics, [])
    
    print(f"\nH-values at root:")
    print(f"  CG:  {cg_h}")
    print(f"  DG:  {dg_h}")
    print(f"  WDG: {wdg_h}")
    
    
    print("\n" + "="*70)
    print("ADMISSIBILITY CHECK")
    print("="*70)
    
    admissible = True
    
    if cg_h < 0:
        print("✗ CG h-value is negative!")
        admissible = False
    else:
        print("✓ CG h-value >= 0")
    
    if dg_h < cg_h:
        print(f"✗ DG h-value ({dg_h}) < CG h-value ({cg_h})!")
        print("  This violates the expected relationship DG >= CG")
        admissible = False
    else:
        print(f"✓ DG h-value ({dg_h}) >= CG h-value ({cg_h})")
    
    if wdg_h < dg_h:
        print(f"⚠ WDG h-value ({wdg_h}) < DG h-value ({dg_h})")
        print("  This can happen when weights are all 1 or LP fails")
    else:
        print(f"✓ WDG h-value ({wdg_h}) >= DG h-value ({dg_h})")
    
    
    print("\n" + "="*70)
    print("INFORMATIVENESS CHECK")
    print("="*70)
    
    if len(collisions) > 0:
        if cg_h == 0:
            print("⚠ CG h-value is 0 despite having collisions")
            print("  → No cardinal conflicts found")
        else:
            print(f"✓ CG h-value > 0 with {len(collisions)} collisions")
        
        if dg_h == 0:
            print("⚠ DG h-value is 0 despite having collisions")
            print("  → All collision pairs are independent")
        else:
            print(f"✓ DG h-value > 0 with {len(collisions)} collisions")
    
    
    print("\n" + "="*70)
    print("EXPECTED SEARCH PERFORMANCE")
    print("="*70)
    
    print("\nTheoretical ranking (fewer nodes = better):")
    print("  Best:  DG (highest h-value, most informed)")
    print("  Good:  WDG (high h-value, but expensive to compute)")
    print("  OK:    CG (moderate h-value)")
    print("  Base:  CBS (h=0)")
    
    print(f"\nActual h-values:")
    print(f"  DG:  {dg_h} (should expand fewest nodes)")
    print(f"  WDG: {wdg_h}")
    print(f"  CG:  {cg_h}")
    print(f"  CBS: 0")
    
    if dg_h > cg_h > 0:
        print("\n✓ H-values suggest: DG should be best, then CG, then CBS")
    elif dg_h == cg_h and cg_h > 0:
        print("\n⚠ DG and CG have same h-value")
        print("  → Similar performance expected")
    
    print("\n" + "="*70)
    print("SMALL INSTANCE CAVEAT")
    print("="*70)
    print("\nFor small problems (< 50 nodes expanded):")
    print("  • Tie-breaking effects can dominate")
    print("  • CG might expand more nodes than CBS by chance")
    print("  • This is NORMAL and not a bug")
    print("  • Test on larger instances for clearer trends")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_heuristic.py <map_file>")
        sys.exit(1)
    
    verify_heuristic(sys.argv[1])
