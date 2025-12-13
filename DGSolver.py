import time as timer
from itertools import combinations
from collections import defaultdict
import copy

from cbs import CBSSolver
from mdd import (
    get_all_optimal_paths,
    buildMDDTree,
    buildJointMDD,
    check_jointMDD_for_dependency,
)
from single_agent_planner import a_star, get_sum_of_cost

class DGSolver(CBSSolver):
    def get_dg_heuristic(self, my_map, paths, starts, goals, low_level_h, constraints):
        num_agents = len(paths)

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

        dependency_graph = defaultdict(list)
        for first_idx, second_idx in agent_pair_dependencies:
            dependency_graph[first_idx].append(second_idx)

        is_in_cover = [False] * num_agents
        for first_agent in range(num_agents):
            if is_in_cover[first_agent]:
                continue
            for second_agent in dependency_graph[first_agent]:
                if is_in_cover[second_agent]:
                    continue
                is_in_cover[second_agent] = True
                is_in_cover[first_agent] = True
                break

        vertex_cover_set = [
            agent_idx for agent_idx in range(num_agents) if is_in_cover[agent_idx]
        ]
        return len(vertex_cover_set)

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
            'h': self.get_dg_heuristic(
                self.my_map,
                base_paths,
                self.starts,
                self.goals,
                self.heuristics,
                base_constraints,
            ),
        }
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
                child_node['h'] = self.get_dg_heuristic(
                    self.my_map,
                    child_node['paths'],
                    self.starts,
                    self.goals,
                    self.heuristics,
                    child_node['constraints'],
                )

                self.push_node(child_node)

        if record_results:
            self.write_results()
            self.print_results(initial_node)
        return initial_node['paths']