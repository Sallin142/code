#!/usr/bin/python
"""
Comprehensive MAPF Solver Benchmarking and Visualization Script
Runs CBS CG, CBS Standard, and CBS Disjoint on benchmark instances
and generates detailed per-test performance graphs with statistical analysis.
"""

import os
import time
import glob
import csv
import argparse
import statistics
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Import MAPF solvers
from cbs import CBSSolver
from CGSolver import CGSolver
from DGSolver import DGSolver
from WDGSolver import WDGSolver
from run_experiments import import_mapf_instance
from single_agent_planner import get_sum_of_cost

class ComprehensiveBenchmark:
    def __init__(self, benchmark_dir='Benchmarks', repeat_count=1):
        self.benchmark_dir = benchmark_dir
        self.repeat_count = repeat_count
        self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # Store lists for multiple runs
        self.solvers = ['CG', 'DG', 'WDG', 'CBS Standard', 'CBS Disjoint']
        self.colors = ['#3498db', '#9b59b6','#f1c40f', '#e74c3c', '#2ecc71']  # Blue, Purple, Yellow, Red, Green
        
    def load_instance(self, filename):
        """Load a MAPF instance from file"""
        try:
            return import_mapf_instance(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None, None, None
    
    def run_solver(self, solver_name, my_map, starts, goals, timeout=30000000000):
        """Run a specific solver and return performance metrics"""
        try:
            start_time = time.time()
            
            if solver_name == 'CG':
                solver = CGSolver(my_map, starts, goals)
                try:
                    paths = solver.find_solution(disjoint=True)
                except BaseException as e:
                    if 'No solutions' in str(e):
                        paths = None
                    else:
                        raise e
                expanded_nodes = getattr(solver, 'num_of_expanded', 0)
                generated_nodes = getattr(solver, 'num_of_generated', 0)
            elif solver_name == 'DG':
                solver = DGSolver( my_map, starts, goals)
                try:
                    paths = solver.find_solution(disjoint = True, record_results = False)
                except BaseException as e:
                    if 'No solutions' in str( e):
                        paths = None
                    else:
                        raise e
                expanded_nodes = getattr(solver, 'num_of_expanded', 0)
                generated_nodes = getattr(solver, 'num_of_generated' , 0)
            elif solver_name == 'WDG':
                solver = WDGSolver( my_map, starts, goals)
                try:
                    paths = solver.find_solution(disjoint = True, record_results = False)
                except BaseException as e:
                    if 'No solutions' in str( e):
                        paths = None
                    else:
                        raise e
                expanded_nodes = getattr(solver, 'num_of_expanded', 0)
                generated_nodes = getattr(solver, 'num_of_generated' , 0)
            elif solver_name == 'CBS Standard':
                solver = CBSSolver(my_map, starts, goals)
                try:
                    paths = solver.find_solution(disjoint=False)
                except BaseException as e:
                    if 'No solutions' in str(e):
                        paths = None
                    else:
                        raise e
                expanded_nodes = getattr(solver, 'num_of_expanded', 0)
                generated_nodes = getattr(solver, 'num_of_generated', 0)
            elif solver_name == 'CBS Disjoint':
                solver = CBSSolver(my_map, starts, goals)
                try:
                    paths = solver.find_solution(disjoint=True)
                except BaseException as e:
                    if 'No solutions' in str(e):
                        paths = None
                    else:
                        raise e
                expanded_nodes = getattr(solver, 'num_of_expanded', 0)
                generated_nodes = getattr(solver, 'num_of_generated', 0)
            else:
                raise ValueError(f"Unknown solver: {solver_name}")
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # Check for timeout
            if runtime > timeout:
                return {
                    'cost': -1,
                    'runtime': runtime,
                    'success': False,
                    'expanded_nodes': 0,
                    'generated_nodes': 0
                }
            
            if paths is None:
                return {
                    'cost': -1,
                    'runtime': runtime,
                    'success': False,
                    'expanded_nodes': expanded_nodes,
                    'generated_nodes': generated_nodes
                }
            
            return {
                'cost': get_sum_of_cost(paths),
                'runtime': runtime,
                'success': True,
                'expanded_nodes': expanded_nodes,
                'generated_nodes': generated_nodes
            }
            
        except Exception as e:
            print(f"Error running {solver_name}: {e}")
            return {
                'cost': -1,
                'runtime': 0,
                'success': False,
                'expanded_nodes': 0,
                'generated_nodes': 0
            }
    
    def discover_test_files(self):
        """Discover all test files in benchmark directory"""
        test_files = {}
        
        for agent_dir in ['max_agents_2','max_agents_10']:
            agent_path = os.path.join(self.benchmark_dir, agent_dir)
            if os.path.exists(agent_path):
                files = []
                for i in range(50):  # test_0 to test_24
                    test_file = os.path.join(agent_path, f'test_{i}.txt')
                    if os.path.exists(test_file):
                        files.append(test_file)
                test_files[agent_dir] = sorted(files)
        
        return test_files
    
    def calculate_statistics(self, results_list, metric):
        """Calculate statistics for a list of results"""
        if not results_list:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0, 'success_rate': 0}
        
        # Filter successful results for most metrics
        if metric == 'success_rate':
            successes = sum(1 for r in results_list if r['success'])
            return {
                'mean': successes / len(results_list) * 100,
                'std': 0,  # Success rate doesn't have std in the same way
                'min': 0 if successes == 0 else 100 if successes == len(results_list) else 50,
                'max': 100 if successes > 0 else 0,
                'count': len(results_list),
                'success_rate': successes / len(results_list) * 100
            }
        
        successful_results = [r for r in results_list if r['success']]
        if not successful_results:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0, 'success_rate': 0}
        
        values = [r[metric] for r in successful_results]
        
        return {
            'mean': statistics.mean(values) if values else 0,
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values) if values else 0,
            'max': max(values) if values else 0,
            'count': len(successful_results),
            'success_rate': len(successful_results) / len(results_list) * 100
        }
    
    def run_all_benchmarks(self):
        """Run all solvers on all benchmark instances with repeated trials"""
        print("=" * 80)
        print("COMPREHENSIVE MAPF SOLVER BENCHMARKING")
        if self.repeat_count > 1:
            print(f"Running {self.repeat_count} iterations per test for statistical analysis")
        print("=" * 80)
        
        test_files = self.discover_test_files()
        
        for agent_group, files in test_files.items():
            print(f"\nProcessing {agent_group} ({len(files)} files)...")
            
            for i, test_file in enumerate(files):
                test_name = os.path.basename(test_file).replace('.txt', '')
                print(f"\n[{i+1}/{len(files)}] Running {test_name}...")
                
                # Load instance
                my_map, starts, goals = self.load_instance(test_file)
                if my_map is None:
                    continue
                
                # Run each solver multiple times
                for solver_name in self.solvers:
                    if self.repeat_count > 1:
                        print(f"  {solver_name} ({self.repeat_count} runs)...", end=' ')
                    else:
                        print(f"  {solver_name}...", end=' ')
                    
                    successes = 0
                    costs = []
                    runtimes = []
                    
                    for run in range(self.repeat_count):
                        result = self.run_solver(solver_name, my_map, starts, goals)
                        
                        if result:
                            self.results[agent_group][test_name][solver_name].append(result)
                            
                            if result['success']:
                                successes += 1
                                costs.append(result['cost'])
                                runtimes.append(result['runtime'])
                    
                    # Print summary for this solver
                    if self.repeat_count > 1:
                        if successes > 0:
                            avg_cost = statistics.mean(costs) if costs else 0
                            avg_time = statistics.mean(runtimes) if runtimes else 0
                            print(f"✓ {successes}/{self.repeat_count} success (avg cost: {avg_cost:.1f}, avg time: {avg_time:.3f}s)")
                        else:
                            print(f"✗ 0/{self.repeat_count} success")
                    else:
                        # Single run - show individual result
                        if self.results[agent_group][test_name][solver_name]:
                            result = self.results[agent_group][test_name][solver_name][0]
                            status = "✓" if result['success'] else "✗"
                            print(f"{status} (cost: {result['cost']}, time: {result['runtime']:.3f}s)")
                        else:
                            print("TIMEOUT")
        
        print(f"\nBenchmarking complete!")
        return self.results
    
    def create_grouped_bar_chart(self, data, title, ylabel, filename, agent_group, log_scale=False):
        """Create a grouped bar chart for averaged performance metrics with error bars"""
        test_names = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x = np.arange(len(test_names))
        width = 0.25
        
        # Determine metric name for statistics calculation
        if ylabel == 'Success Rate':
            metric = 'success_rate'
        elif 'cost' in ylabel.lower():
            metric = 'cost'
        elif 'runtime' in ylabel.lower():
            metric = 'runtime'
        elif 'expanded' in ylabel.lower():
            metric = 'expanded_nodes'
        elif 'generated' in ylabel.lower():
            metric = 'generated_nodes'
        else:
            metric = 'cost'
        
        for i, solver in enumerate(self.solvers):
            values = []
            errors = []
            
            for test in test_names:
                if solver in data[test] and data[test][solver]:
                    stats = self.calculate_statistics(data[test][solver], metric)
                    values.append(stats['mean'])
                    errors.append(stats['std'])
                else:
                    values.append(0)
                    errors.append(0)
            
            bars = ax.bar(x + i * width, values, width, yerr=errors if self.repeat_count > 1 else None, 
                         label=solver, color=self.colors[i], alpha=0.8, 
                         capsize=5 if self.repeat_count > 1 else 0, 
                         error_kw={'linewidth': 1, 'alpha': 0.7} if self.repeat_count > 1 else {})
            
            # Add value labels on bars for smaller datasets
            if len(test_names) <= 15:
                for j, (bar, value, error) in enumerate(zip(bars, values, errors)):
                    if value > 0:
                        height = bar.get_height()
                        if self.repeat_count > 1 and error > 0.01:
                            label_text = f'{value:.1f}±{error:.1f}'
                        else:
                            label_text = f'{value:.1f}'
                        y_offset = error if self.repeat_count > 1 else 0
                        ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                               label_text, ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('Test Instance')
        ax.set_ylabel(ylabel)
        
        title_suffix = f" - {agent_group.replace('max_agents_', '').replace('_', ' ')} Agents"
        if self.repeat_count > 1:
            title_suffix += f" (Averaged over {self.repeat_count} runs)"
        ax.set_title(f'{title}{title_suffix}')
        
        ax.set_xticks(x + width)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            # Only use log scale if we have values > 1
            max_value = max(max(values) if values else 0, 1)
            if max_value > 10:
                ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
    
    def create_cbs_comparison_chart(self, data, agent_group):
        """Create CBS Standard vs Disjoint expanded nodes comparison with error bars"""
        test_names = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x = np.arange(len(test_names))
        width = 0.35
        
        standard_values = []
        standard_errors = []
        disjoint_values = []
        disjoint_errors = []
        
        for test in test_names:
            # CBS Standard statistics
            if 'CBS Standard' in data[test] and data[test]['CBS Standard']:
                std_stats = self.calculate_statistics(data[test]['CBS Standard'], 'expanded_nodes')
                standard_values.append(std_stats['mean'])
                standard_errors.append(std_stats['std'])
            else:
                standard_values.append(0)
                standard_errors.append(0)
            
            # CBS Disjoint statistics
            if 'CBS Disjoint' in data[test] and data[test]['CBS Disjoint']:
                dis_stats = self.calculate_statistics(data[test]['CBS Disjoint'], 'expanded_nodes')
                disjoint_values.append(dis_stats['mean'])
                disjoint_errors.append(dis_stats['std'])
            else:
                disjoint_values.append(0)
                disjoint_errors.append(0)
        
        bars1 = ax.bar(x - width/2, standard_values, width, 
                      yerr=standard_errors if self.repeat_count > 1 else None,
                      label='CBS Standard', color='#e74c3c', alpha=0.8, 
                      capsize=5 if self.repeat_count > 1 else 0, 
                      error_kw={'linewidth': 1, 'alpha': 0.7} if self.repeat_count > 1 else {})
        bars2 = ax.bar(x + width/2, disjoint_values, width, 
                      yerr=disjoint_errors if self.repeat_count > 1 else None,
                      label='CBS Disjoint', color='#2ecc71', alpha=0.8,
                      capsize=5 if self.repeat_count > 1 else 0, 
                      error_kw={'linewidth': 1, 'alpha': 0.7} if self.repeat_count > 1 else {})
        
        # Add value labels
        if len(test_names) <= 15:
            for values, errors, bars in [(standard_values, standard_errors, bars1), 
                                       (disjoint_values, disjoint_errors, bars2)]:
                for bar, value, error in zip(bars, values, errors):
                    if value > 0:
                        height = bar.get_height()
                        if self.repeat_count > 1 and error > 0.1:
                            label_text = f'{value:.0f}±{error:.0f}'
                        else:
                            label_text = f'{value:.0f}'
                        y_offset = error if self.repeat_count > 1 else 0
                        ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                               label_text, ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('Test Instance')
        ax.set_ylabel('Expanded Nodes')
        
        title = f'CBS Standard vs Disjoint: Expanded Nodes Comparison - {agent_group.replace("max_agents_", "").replace("_", " ")} Agents'
        if self.repeat_count > 1:
            title += f' (Averaged over {self.repeat_count} runs)'
        ax.set_title(title)
        
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Use log scale if values are large
        if standard_values or disjoint_values:
            combined_values = standard_values + disjoint_values
            if combined_values:
                max_value = max(combined_values)
                if max_value > 100:
                    ax.set_yscale('log')
        
        plt.tight_layout()
        filename = f'cbs_comparison_{agent_group.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
    
    def generate_all_graphs(self):
        """Generate all performance comparison graphs"""
        print("\n" + "=" * 80)
        print("GENERATING PERFORMANCE GRAPHS")
        print("=" * 80)
        
        metrics = [
            ('Cost', 'Sum of Costs', False),
            ('Runtime', 'Runtime (seconds)', False),
            ('Success Rate', 'Success Rate', False),
            ('Expanded Nodes', 'Expanded Nodes', True),
            ('Generated Nodes', 'Generated Nodes', True)
        ]
        
        for agent_group in self.results:
            print(f"\nGenerating graphs for {agent_group}...")
            data = self.results[agent_group]
            
            # Generate individual metric graphs
            for metric, ylabel, log_scale in metrics:
                filename = f'{metric.lower().replace(" ", "_")}_{agent_group.lower()}.png'
                self.create_grouped_bar_chart(data, metric, ylabel, filename, agent_group, log_scale)
            
            # Generate CBS comparison graph
            self.create_cbs_comparison_chart(data, agent_group)
    
    def save_results_csv(self):
        """Save detailed results to CSV files with statistical summaries"""
        print("\nSaving results to CSV files...")
        
        for agent_group, data in self.results.items():
            # Save individual run data
            if self.repeat_count > 1:
                filename = f'benchmark_results_raw_{agent_group.lower()}.csv'
                with open(filename, 'w', newline='') as csvfile:
                    fieldnames = ['test_name', 'solver', 'run', 'success', 'cost', 'runtime', 
                                 'expanded_nodes', 'generated_nodes']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for test_name in sorted(data.keys(), key=lambda x: int(x.split('_')[1])):
                        for solver in self.solvers:
                            if solver in data[test_name]:
                                for run_idx, result in enumerate(data[test_name][solver]):
                                    writer.writerow({
                                        'test_name': test_name,
                                        'solver': solver,
                                        'run': run_idx + 1,
                                        'success': result['success'],
                                        'cost': result['cost'] if result['success'] else -1,
                                        'runtime': result['runtime'],
                                        'expanded_nodes': result['expanded_nodes'],
                                        'generated_nodes': result['generated_nodes']
                                    })
                print(f"Saved {filename}")
            
            # Save statistical summary
            filename = f'benchmark_results_summary_{agent_group.lower()}.csv'
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['test_name', 'solver', 'success_rate', 'total_runs',
                             'cost_mean', 'cost_std', 'cost_min', 'cost_max',
                             'runtime_mean', 'runtime_std', 'runtime_min', 'runtime_max',
                             'expanded_nodes_mean', 'expanded_nodes_std', 'expanded_nodes_min', 'expanded_nodes_max',
                             'generated_nodes_mean', 'generated_nodes_std', 'generated_nodes_min', 'generated_nodes_max']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for test_name in sorted(data.keys(), key=lambda x: int(x.split('_')[1])):
                    for solver in self.solvers:
                        if solver in data[test_name] and data[test_name][solver]:
                            results = data[test_name][solver]
                            
                            cost_stats = self.calculate_statistics(results, 'cost')
                            runtime_stats = self.calculate_statistics(results, 'runtime')
                            expanded_stats = self.calculate_statistics(results, 'expanded_nodes')
                            generated_stats = self.calculate_statistics(results, 'generated_nodes')
                            success_stats = self.calculate_statistics(results, 'success_rate')
                            
                            writer.writerow({
                                'test_name': test_name,
                                'solver': solver,
                                'success_rate': success_stats['mean'],
                                'total_runs': len(results),
                                'cost_mean': cost_stats['mean'],
                                'cost_std': cost_stats['std'],
                                'cost_min': cost_stats['min'],
                                'cost_max': cost_stats['max'],
                                'runtime_mean': runtime_stats['mean'],
                                'runtime_std': runtime_stats['std'],
                                'runtime_min': runtime_stats['min'],
                                'runtime_max': runtime_stats['max'],
                                'expanded_nodes_mean': expanded_stats['mean'],
                                'expanded_nodes_std': expanded_stats['std'],
                                'expanded_nodes_min': expanded_stats['min'],
                                'expanded_nodes_max': expanded_stats['max'],
                                'generated_nodes_mean': generated_stats['mean'],
                                'generated_nodes_std': generated_stats['std'],
                                'generated_nodes_min': generated_stats['min'],
                                'generated_nodes_max': generated_stats['max']
                            })
            
            print(f"Saved {filename}")
    
    def print_summary_statistics(self):
        """Print summary statistics"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY STATISTICS")
        print("=" * 80)
        
        for agent_group, data in self.results.items():
            print(f"\n{agent_group.replace('max_agents_', '').replace('_', ' ')} Agents:")
            print("-" * 40)
            
            total_tests = len(data)
            
            for solver in self.solvers:
                # Collect all results for this solver across all tests
                all_solver_results = []
                tests_with_results = 0
                
                for test_data in data.values():
                    if solver in test_data and test_data[solver]:
                        tests_with_results += 1
                        all_solver_results.extend(test_data[solver])
                
                if all_solver_results:
                    # Calculate statistics using the results lists
                    cost_stats = self.calculate_statistics(all_solver_results, 'cost')
                    runtime_stats = self.calculate_statistics(all_solver_results, 'runtime')
                    expanded_stats = self.calculate_statistics(all_solver_results, 'expanded_nodes')
                    generated_stats = self.calculate_statistics(all_solver_results, 'generated_nodes')
                    success_stats = self.calculate_statistics(all_solver_results, 'success_rate')
                    
                    total_runs = len(all_solver_results)
                    successful_runs = sum(1 for r in all_solver_results if r['success'])
                    
                    print(f"{solver}:")
                    print(f"  Tests with results: {tests_with_results}/{total_tests}")
                    print(f"  Total runs: {total_runs}")
                    print(f"  Success rate: {success_stats['mean']:.1f}% ({successful_runs}/{total_runs})")
                    print(f"  Avg Cost: {cost_stats['mean']:.1f} ± {cost_stats['std']:.1f}")
                    print(f"  Avg Runtime: {runtime_stats['mean']:.3f}s ± {runtime_stats['std']:.3f}s")
                    print(f"  Avg Expanded Nodes: {expanded_stats['mean']:.1f} ± {expanded_stats['std']:.1f}")
                    print(f"  Avg Generated Nodes: {generated_stats['mean']:.1f} ± {generated_stats['std']:.1f}")
                else:
                    print(f"{solver}: No results")
                print()

def main():
    parser = argparse.ArgumentParser(description='Comprehensive MAPF Solver Benchmarking with Statistical Analysis')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of times to repeat each test (default: 1)')
    parser.add_argument('--benchmark_dir', type=str, default='custominstances/Benchmarks',
                        help='Directory containing benchmark instances (default: Benchmarks)')
    
    args = parser.parse_args()
    
    if args.repeat < 1:
        print("Error: --repeat must be at least 1")
        return
    
    # Create benchmark instance
    benchmark = ComprehensiveBenchmark(benchmark_dir=args.benchmark_dir, repeat_count=args.repeat)
    
    # Run all benchmarks
    benchmark.run_all_benchmarks()
    
    # Generate graphs
    benchmark.generate_all_graphs()
    
    # Save CSV results
    benchmark.save_results_csv()
    
    # Print summary
    benchmark.print_summary_statistics()
    
    print("\n" + "=" * 80)
    print("BENCHMARKING COMPLETE!")
    if args.repeat > 1:
        print(f"Statistical analysis based on {args.repeat} runs per test")
    print("Generated graphs:")
    print("- Cost comparison charts (per agent group)" + (" with error bars" if args.repeat > 1 else ""))
    print("- Runtime comparison charts (per agent group)" + (" with error bars" if args.repeat > 1 else ""))
    print("- Success rate charts (per agent group)")
    print("- Expanded nodes charts (per agent group)" + (" with error bars" if args.repeat > 1 else ""))
    print("- Generated nodes charts (per agent group)" + (" with error bars" if args.repeat > 1 else ""))
    print("- CBS Standard vs Disjoint comparison charts" + (" with error bars" if args.repeat > 1 else ""))
    print("- Detailed CSV results files with statistical summaries")
    print("=" * 80)

if __name__ == '__main__':
    main()