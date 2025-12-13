import time as timer
from itertools import combinations
import copy

import pulp as pl
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize, value

from cbs import CBSSolver
from CGSolver import CGSolver
from mdd import (
    check_jointMDD_for_dependency,
    buildJointMDD,
    buildMDDTree,
    get_all_optimal_paths,
)
from single_agent_planner import a_star, get_sum_of_cost

class WDGSolver(CBSSolver):
    def get_wdg_heuristic(self, my_map, paths, starts, goals, low_level_h, constraints):
        num_agents = len(paths)
        lp_model = LpProblem("edge_weighted_minimum_vertex_cover", LpMinimize)

        def build_agent_data():
            agent_data = []
            for agent_idx in range(num_agents):
                optimal_paths = get_all_optimal_paths(
                    my_map,
                    starts[agent_idx],
                    goals[agent_idx],
                    low_level_h[agent_idx],
                    agent_idx,
                    constraints,
                )
                mdd_root, node_dictionary = buildMDDTree(optimal_paths)
                agent_data.append(
                    {
                        "idx": agent_idx,
                        "paths": optimal_paths,
                        "root": mdd_root,
                        "nodes": node_dictionary,
                    }
                )
            return agent_data

        agent_info = build_agent_data()
        agent_info_by_idx = {item["idx"]: item for item in agent_info}
        agent_pair_dependencies = []

        for first_info, second_info in combinations(agent_info, 2):
            joint_mdd_root, joint_bottom_node = buildJointMDD(
                first_info["paths"],
                second_info["paths"],
                first_info["root"],
                first_info["nodes"],
                second_info["root"],
                second_info["nodes"],
            )
            if not check_jointMDD_for_dependency(
                joint_bottom_node, first_info["paths"], second_info["paths"]
            ):
                continue
            agent_pair_dependencies.append((first_info["idx"], second_info["idx"]))

        lp_variables_by_agent = {}

        def get_lp_var(agent_idx):
            if agent_idx not in lp_variables_by_agent:
                lp_variables_by_agent[agent_idx] = LpVariable(
                    f"a{agent_idx}", lowBound=0, cat="Integer", e=None
                )
            return lp_variables_by_agent[agent_idx]

        for first_agent_idx, second_agent_idx in agent_pair_dependencies:
            first_var = get_lp_var(first_agent_idx)
            second_var = get_lp_var(second_agent_idx)

            pairwise_starts = [starts[first_agent_idx], starts[second_agent_idx]]
            pairwise_goals = [goals[first_agent_idx], goals[second_agent_idx]]

            pairwise_solver = CGSolver(my_map, pairwise_starts, pairwise_goals)
            joint_solution_paths = pairwise_solver.find_solution(record_results=False)
            if joint_solution_paths == []:
                return -1
            joint_optimal_cost = get_sum_of_cost(joint_solution_paths)

            first_data = agent_info_by_idx[first_agent_idx]
            second_data = agent_info_by_idx[second_agent_idx]
            individual_path_lengths = len(first_data["paths"][0]) + len(
                second_data["paths"][0]
            )
            conflict_cost = individual_path_lengths - joint_optimal_cost
            lp_model += first_var + second_var >= conflict_cost

        objective_function = lpSum(lp_variables_by_agent.values()) if lp_variables_by_agent else 0
        lp_model += objective_function
        lp_model.solve(pl.PULP_CBC_CMD(msg=False))

        heuristic_value = value(lp_model.objective)
        return heuristic_value

    def find_solution(self, disjoint=True, record_results=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()
        base_constraints = []
        base_paths = []
        for agent_idx in range(self.num_of_agents):
            initial_path = a_star(
                self.my_map,
                self.starts[agent_idx],
                self.goals[agent_idx],
                self.heuristics[agent_idx],
                agent_idx,
                base_constraints,
            )
            if initial_path is None:
                raise BaseException('No solutions')
            base_paths.append(initial_path)

        initial_node = {
            'constraints': base_constraints,
            'paths': base_paths,
            'collisions': [],
            'cost': get_sum_of_cost(base_paths),
            'h': self.get_wdg_heuristic(
                self.my_map,
                base_paths,
                self.starts,
                self.goals,
                self.heuristics,
                base_constraints,
            ),
        }
        if initial_node['h'] == -1:
            raise BaseException('No solution')

        initial_node['collisions'] = super().detect_collisions(initial_node['paths'])
        self.push_node(initial_node)

        while self.open_list:
            current_node = self.pop_node()

            if not current_node['collisions']:
                if record_results:
                    self.print_results(current_node)
                    self.write_results()
                return current_node['paths']

            first_collision = current_node['collisions'][0]
            generated_constraints = super().disjoint_splitting(first_collision)

            for new_constraint in generated_constraints:
                if super().is_conflicting_constraint(new_constraint, current_node['constraints']):
                    continue

                child_node = {
                    'constraints': copy.deepcopy(current_node['constraints']),
                    'paths': copy.deepcopy(current_node['paths']),
                }
                if new_constraint not in child_node['constraints']:
                    child_node['constraints'].append(new_constraint)

                should_prune = False
                if new_constraint['positive']:
                    affected_agents = super().paths_violate_constraint(new_constraint, child_node['paths'])
                    for affected_idx in affected_agents:
                        updated_path = a_star(
                            self.my_map,
                            self.starts[affected_idx],
                            self.goals[affected_idx],
                            self.heuristics[affected_idx],
                            affected_idx,
                            child_node['constraints'],
                        )
                        if updated_path is None:
                            should_prune = True
                            break
                        child_node['paths'][affected_idx] = updated_path
                if should_prune:
                    continue

                constrained_agent = new_constraint['agent']
                replanned_path = a_star(
                    self.my_map,
                    self.starts[constrained_agent],
                    self.goals[constrained_agent],
                    self.heuristics[constrained_agent],
                    constrained_agent,
                    child_node['constraints'],
                )
                if replanned_path is None:
                    continue

                child_node['paths'][constrained_agent] = replanned_path
                child_node['collisions'] = super().detect_collisions(child_node['paths'])
                child_node['cost'] = get_sum_of_cost(child_node['paths'])
                child_node['h'] = self.get_wdg_heuristic(
                    self.my_map,
                    child_node['paths'],
                    self.starts,
                    self.goals,
                    self.heuristics,
                    child_node['constraints'],
                )

                if child_node['h'] != -1:
                    self.push_node(child_node)

        if record_results:
            self.print_results(initial_node)
            self.write_results()
        return initial_node['paths']