#!/usr/bin/python
"""
Comprehensive MAPF Benchmark Script which calculates Aaerage h-values, runtime, expanded Nodes, generated Nodes

Usage:
    python real-map-benchmark.py --map empty-map --agents 5 10 20
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO
from cbs import CBSSolver
from CGSolver import CGSolver
from DGSolver import DGSolver
from WDGSolver import WDGSolver
from single_agent_planner import a_star, get_sum_of_cost

# suppress print statements
class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb ):
        sys.stdout = self._original_stdout

def import_mapf_instance(filename):
    """Import a MAPF instance from a file"""
    f = Path(filename)
    if not f.is_file():
        raise BaseException(filename + " does not exist.")
    f = open(filename, 'r')
    
    line = f.readline()
    rows, columns = [int(x) for x in line.split(' ')]
    my_map = []
    for r in range(rows):
        line = f.readline()
        my_map.append([])
        for cell in line:
            if cell == '@':
                my_map[-1].append(True)
            elif cell == '.':
                my_map[-1].append(False)
    
    # agents
    line = f.readline()
    num_agents = int(line)
    
    # agents lines with the start/goal positions
    starts = []
    goals = []
    for a in range(num_agents):
        line = f.readline()
        sx, sy, gx, gy = [int(x) for x in line.split(' ') ]
        starts.append((sx, sy))
        goals.append((gx, gy ))
    
    f.close()
    return my_map, starts, goals

def run_solver(solver_name, my_map, starts, goals, timeout = 60):
    try:
        # create solver instance
        if solver_name == "CBS":
            solver = CBSSolver(my_map, starts, goals)
        elif solver_name == "CG":
            solver = CGSolver(my_map, starts, goals)
        elif solver_name == "DG":
            solver = DGSolver(my_map, starts, goals)
        elif solver_name == "WDG":
            solver = WDGSolver(my_map, starts, goals)
        else:
            raise ValueError(f"Unknown solver: {solver_name}" )
        
        # find solution
        with SuppressPrints():
            paths = solver.find_solution(disjoint=True)
        
        # check if solution found
        if paths is None or len(paths) == 0:
            return {
                'success': False,
                'cpu_time': timeout,
                'num_expanded': -1,
                'num_generated': -1,
                'cost': -1,
                'root_h': -1
            }
        
        # extract results
        result = {
            'success': True,
            'cpu_time': solver.CPU_time,
            'num_expanded': solver.num_of_expanded,
            'num_generated': solver.num_of_generated,
            'cost': get_sum_of_cost(paths),
            'root_h': getattr(solver, 'root_h_value', 0)
        }
        
        return result
        
    except Exception as e:
        print(f"      ERROR: {e}")
        return {
            'success': False,
            'cpu_time': timeout,
            'num_expanded': -1,
            'num_generated': -1,
            'cost': -1,
            'root_h': -1
        }

def get_root_h_value(solver_name, my_map, starts, goals):
    try:
        # Create solver
        if solver_name == "CBS":
            return 0
        elif solver_name == "CG":
            solver = CGSolver(my_map, starts, goals)
            method = solver.get_cg_heuristic
        elif solver_name == "DG":
            solver = DGSolver(my_map, starts, goals)
            method = solver.get_dg_heuristic
        elif solver_name == "WDG":
            solver = WDGSolver(my_map, starts, goals)
            method = solver.get_wdg_heuristic
        else:
            return -1
        
        # get initial paths
        paths = []
        for i in range(len(starts)):
            path = a_star(my_map, starts[i], goals[i], solver.heuristics[i], i, [])
            if path is None:
                return -1
            paths.append(path)
        
        # compute h-value
        h_val = method(my_map, paths, starts, goals, solver.heuristics, [])
        return h_val if h_val != -1 else 0
        
    except Exception as e:
        print(f"ERROR computing h-value: {e}")
        return -1

# process all instances for a specific map type and agent count
def process_agent_folder(map_name, num_agents, solvers, base_path = "real-map-instance"):

    # construct folder path
    map_prefix = map_name.split('-')[0]
    folder_name = f"{map_prefix}-32-32-{num_agents}agents"
    folder_path = os.path.join(base_path, map_name , folder_name )
    
    if not os.path.exists(folder_path):
        print(f"WARNING: Folder not found: {folder_path}")
        return None
    
    print(f"\n  Processing {num_agents} agents...")
    print(f"Folder: {folder_path}")
    
    # get all instance files
    instance_files = sorted(glob.glob(os.path.join( folder_path, "*.txt")))
    
    if len(instance_files) == 0:
        print(f"No instance files found!")
        return None
    
    print(f"  Found {len(instance_files)} instances")
    print(f"  {'-'*70}")
    
    # store results for each solver
    solver_results = {solver: {
        'h_values': [],
        'cpu_times': [],
        'num_expanded': [],
        'num_generated': [],
        'costs': [],
        'successes': 0,
        'total': 0
    } for solver in solvers}
    
    # process each instance
    for idx, instance_file in enumerate(instance_files, 1):
        instance_name = os.path.basename(instance_file)
        print(f"  [{idx:2d}/{len(instance_files)}] {instance_name}")
        
        # load instance
        my_map, starts, goals = import_mapf_instance( instance_file)
        
        # run each solver
        for solver_name in solvers:
            print(f"{solver_name}...", end=' ')
            
            result = run_solver(solver_name, my_map, starts, goals)
            
            # extract h-value from result
            h_val = result.get('root_h', 0)
            
            # store results
            solver_results[solver_name]['total'] += 1
            
            if result['success']:
                solver_results[solver_name]['successes'] += 1
                solver_results[solver_name]['h_values'].append(h_val)
                solver_results[solver_name]['cpu_times'].append(result['cpu_time'])
                solver_results[solver_name]['num_expanded'].append(result['num_expanded'])
                solver_results[solver_name]['num_generated'].append(result['num_generated'])
                solver_results[solver_name]['costs'].append(result['cost'])
                
                print(f"h={h_val:.1f}, t={result['cpu_time']:.3f}s, " +
                      f"exp={result['num_expanded']}, gen={result['num_generated']}")
            else:
                print(f"FAILED")
    
    # compute averages
    print(f"\n  Summary for {num_agents} agents:")
    print(f"  {'-'*70}")
    
    results_row = {'num_agents': num_agents, 'instances': len(instance_files)}
    
    for solver_name in solvers:
        sr = solver_results[solver_name]
        success_count = sr['successes']
        
        if success_count > 0:
            results_row[f'{solver_name}_instances_solved'] = success_count
            results_row[f'{solver_name}_h_avg'] = np.mean(sr['h_values'])
            results_row[f'{solver_name}_time_avg'] = np.mean(sr['cpu_times'])
            results_row[f'{solver_name}_expanded_avg'] = np.mean(sr['num_expanded'])
            results_row[f'{solver_name}_generated_avg'] = np.mean(sr['num_generated'])
            
            print(f"  {solver_name:4s}: {success_count:2d}/{sr['total']:2d} solved | " +
                  f"h={results_row[f'{solver_name}_h_avg']:5.1f} | " +
                  f"time={results_row[f'{solver_name}_time_avg']:7.3f}s | " +
                  f"expanded={results_row[f'{solver_name}_expanded_avg']:8.0f} | " +
                  f"generated={results_row[f'{solver_name}_generated_avg']:9.0f}")
        else:
            print(f"  {solver_name:4s}: 0/{sr['total']} solved (all failed)")
            results_row[f'{solver_name}_instances_solved'] = 0
            results_row[f'{solver_name}_h_avg'] = 0
            results_row[f'{solver_name}_time_avg'] = 0
            results_row[f'{solver_name}_expanded_avg'] = 0
            results_row[f'{solver_name}_generated_avg'] = 0
    
    return results_row

# create comprehensive results table
def create_results_table(all_results, solvers, map_name, output_folder):

    if not all_results:
        print("No results to create table!")
        return None
    
    df = pd.DataFrame(all_results)
    df = df.sort_values('num_agents')
    
    # Save resusts in CSV
    csv_file = os.path.join(output_folder, f'{map_name}_full_results.csv' )
    df.to_csv(csv_file, index=False)
    print(f"\nFull results saved to: {csv_file}")
    
    # create summary table
    summary_rows = []
    
    for _, row in df.iterrows():
        k = row['num_agents']
        instances = row['instances']
        
        summary_row = {'Agents': k, 'Instances': instances}
        for solver in solvers:
            solved = row.get(f'{solver}_instances_solved', 0)
            if solved > 0:
                summary_row[f'{solver}_Nodes(×1000)'] = row[f'{solver}_expanded_avg'] / 1000
                summary_row[f'{solver}_Runtime(s)'] = row[f'{solver}_time_avg']
            else:
                summary_row[f'{solver}_Nodes(×1000)'] = '-'
                summary_row[f'{solver}_Runtime(s)'] = '-'
        
        summary_rows.append(summary_row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(output_folder, f'{map_name}_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary table saved to: {summary_file}")
    
    return df

# plot
def create_plots(df, solvers, map_name, output_folder):
    if df is None or df.empty:
        return
    
    # Plot 1: H-values comparison (Bar chart)
    if all(f'{solver}_h_avg' in df.columns for solver in solvers if solver != 'CBS'):
        fig, ax = plt.subplots(figsize=( 12, 6))
        
        x = np.arange(len(df))
        width = 0.25
        
        heuristic_solvers = [s for s in solvers if s != 'CBS']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, solver in enumerate(heuristic_solvers):
            offset = width * (i - len(heuristic_solvers)/2 + 0.5)
            bars = ax.bar(x + offset, df[f'{solver}_h_avg'], width, 
                         label = solver, alpha = 0.8, color = colors[i])
            
            # add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha = 'center', va = 'bottom', fontsize = 9)
        
        ax.set_xlabel('Number of Agents', fontsize = 12, fontweight = 'bold')
        ax.set_ylabel('Average h-value of Root CT Node', fontsize = 12, fontweight = 'bold')
        ax.set_title(f'{map_name.replace("-", " ").title()}: Root Node H-values', 
                    fontsize = 14, fontweight = 'bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['num_agents'].astype(int ))
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha = 0.3, linestyle = '--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{map_name}_h_values.png'), dpi = 300, bbox_inches = 'tight')
        print(f"Saved: {map_name}_h_values.png")
        plt.close()
    
    # Plot 2: Runtime comparison (Line chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    markers = ['o', 's', '^', 'D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, solver in enumerate(solvers):
        col = f'{solver}_time_avg'
        if col in df.columns:
            mask = df[f'{solver}_instances_solved'] > 0
            if mask.any():
                ax.plot(df[mask]['num_agents'], df[mask][col], 
                       marker = markers[i], label = solver, linewidth = 2.5, 
                       markersize = 10, color = colors[i])
    
    ax.set_xlabel('Number of Agents', fontsize = 12, fontweight = 'bold')
    ax.set_ylabel('Average Runtime (seconds)', fontsize = 12 , fontweight='bold')
    ax.set_title(f'{map_name.replace("-", " ").title()}: Runtime Comparison', 
                fontsize = 14, fontweight = 'bold')
    ax.legend(fontsize = 11)
    ax.grid(True, alpha = 0.3, linestyle = '--')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{map_name}_runtime.png'), 
               dpi = 300, bbox_inches = 'tight')
    print(f"Saved: {map_name}_runtime.png")
    plt.close()
    
    # Plot 3: expanded nodes comparison Line chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, solver in enumerate(solvers):
        col = f'{solver}_expanded_avg'
        if col in df.columns:
            mask = df[f'{solver}_instances_solved'] > 0
            if mask.any():
                ax.plot(df[mask]['num_agents'], df[mask][col] / 1000, 
                       marker = markers[i], label = solver, linewidth = 2.5, 
                       markersize = 10, color = colors[i])
    
    ax.set_xlabel('Number of Agents', fontsize = 12, fontweight = 'bold')
    ax.set_ylabel('Average Expanded Nodes (×1000)', fontsize = 12, fontweight = 'bold')
    ax.set_title(f'{map_name.replace("-", " ").title()}: Expanded Nodes Comparison',fontsize = 14, fontweight = 'bold')
    ax.legend(fontsize = 11)
    ax.grid(True, alpha = 0.3, linestyle='--')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{map_name}_expanded_nodes.png'), 
               dpi=300, bbox_inches='tight')
    print(f"Saved: {map_name}_expanded_nodes.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='MAPF Comprehensive Benchmark')
    parser.add_argument('--map', type = str, required = True, help = 'Map type: empty-map, dense-map, or maze-map')
    parser.add_argument('--agents', nargs = '+', type = int, required = True, help = 'Agent counts to test (e.g., 10 20 30)')
    parser.add_argument('--solvers', nargs = '+', default = ['CBS', 'CG', 'DG', 'WDG'], help = 'Solvers to test (default: CBS CG DG WDG)')
    parser.add_argument('--base-path', type = str, default = 'real-map-instance', help = 'Base path to map instances folder')
    parser.add_argument('--output', type = str, default = 'real_map_benchmark_results',help = 'Output folder for results')
    
    args = parser.parse_args()
    
    # validate inputs
    map_name = args.map
    agent_counts = sorted( args.agents)
    solvers = args.solvers
    
    print("="*80)
    print(f"MAPF BENCHMARK: {map_name.upper()}")
    print("="*80)
    print(f"Agent counts: {agent_counts}")
    print(f"Solvers: {solvers}")
    print(f"Base path: {args.base_path}")
    print("="*80)
    
    # create output folder
    os.makedirs(args.output, exist_ok = True)
    
    # process each agent count
    all_results = []
    
    for num_agents in agent_counts:
        result_row = process_agent_folder(map_name, num_agents, solvers, args.base_path)
        if result_row is not None:
            all_results.append( result_row)
    
    # create results table and plots
    if all_results:

        print("Creating resutls table and plots... ")
        
        df = create_results_table(all_results, solvers, map_name, args.output)
        create_plots(df, solvers, map_name, args.output )
        
        print("\n" + "="*80)
        print("Bench mark finished!!!")
        print("="*80)
    else:
        print("\nNo results were generated!")

if __name__ == '__main__':
    main()
