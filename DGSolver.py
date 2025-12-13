import time as timer
from single_agent_planner import a_star, get_sum_of_cost
from cbs import CBSSolver
import copy
from mdd import get_all_optimal_paths, buildMDDTree, buildJointMDD, check_jointMDD_for_dependency
from collections import defaultdict

class DGSolver(CBSSolver):
    def get_dg_heuristic(self, my_map, paths, starts, goals, low_level_h, constraints):
        agent_pair_dependencies = []

        mdd_root_nodes = []
        optimal_path_sets = []
        mdd_node_dicts = []
        for agent_idx in range(len(paths)):
            optimal_paths = get_all_optimal_paths(my_map, starts[agent_idx], goals[agent_idx], low_level_h[agent_idx], agent_idx, constraints)
            mdd_root, node_dictionary = buildMDDTree(optimal_paths)
            mdd_root_nodes.append(mdd_root)
            optimal_path_sets.append(optimal_paths)
            mdd_node_dicts.append(node_dictionary)

        for first_idx in range(len(paths)): 
            
            
            first_agent_paths = optimal_path_sets[first_idx]
            first_mdd_root = mdd_root_nodes[first_idx]
            first_mdd_nodes = mdd_node_dicts[first_idx]

            for second_idx in range(first_idx+1,len(paths)):
                
                
                second_agent_paths = optimal_path_sets[second_idx]
                second_mdd_root = mdd_root_nodes[second_idx]
                second_mdd_nodes = mdd_node_dicts[second_idx]
                
                joint_mdd_root, joint_bottom_node = buildJointMDD(first_agent_paths, second_agent_paths, first_mdd_root, first_mdd_nodes, second_mdd_root, second_mdd_nodes)

                if (check_jointMDD_for_dependency(joint_bottom_node, first_agent_paths, second_agent_paths)):
                    agent_pair_dependencies.append((first_idx,second_idx))

        # Build graph and compute vertex cover inline
        dependency_graph = defaultdict(list)
        for agent_pair in agent_pair_dependencies:
            dependency_graph[agent_pair[0]].append(agent_pair[1])
        
        # Compute vertex cover
        is_in_cover = [False] * len(paths)
        for first_agent in range(len(paths)):
            if not is_in_cover[first_agent]:
                for second_agent in dependency_graph[first_agent]:
                    if not is_in_cover[second_agent]:
                        is_in_cover[second_agent] = True
                        is_in_cover[first_agent] = True
                        break
        
        vertex_cover_set = []
        for agent_idx in range(len(paths)):
            if is_in_cover[agent_idx]:
                vertex_cover_set.append(agent_idx)
        
        return len(vertex_cover_set)

    def find_solution(self, disjoint=True, record_results=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        initial_node = {'cost': 0,
                'h': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for agent_idx in range(self.num_of_agents):  
            initial_path = a_star(self.my_map, self.starts[agent_idx], self.goals[agent_idx], self.heuristics[agent_idx],
                          agent_idx, initial_node['constraints'])
            if initial_path is None:
                raise BaseException('No solutions')
            initial_node['paths'].append(initial_path)

        initial_node['cost'] = get_sum_of_cost(initial_node['paths'])
        initial_node['h'] = self.get_dg_heuristic(self.my_map, initial_node['paths'], self.starts, self.goals, self.heuristics, initial_node['constraints']) 
        initial_node['collisions'] = super().detect_collisions(initial_node['paths'])
        self.push_node(initial_node)

        while len(self.open_list) > 0:
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
                child_node = {}
                child_node['constraints'] = copy.deepcopy(current_node['constraints'])
                if new_constraint not in child_node['constraints']:
                    child_node['constraints'].append(new_constraint)
                child_node['paths']= copy.deepcopy(current_node['paths'])

                should_prune = False
                if new_constraint['positive']:
                    affected_agents = super().paths_violate_constraint(new_constraint, child_node['paths'])
                    for affected_idx in affected_agents:
                        updated_path = a_star(self.my_map, self.starts[affected_idx], self.goals[affected_idx], self.heuristics[affected_idx],
                            affected_idx, child_node['constraints'])
                        if updated_path is None:
                            should_prune = True
                            break
                        else:
                            child_node['paths'][affected_idx] = updated_path
                if should_prune:
                    continue

                constrained_agent = new_constraint['agent']
                replanned_path = a_star(self.my_map, self.starts[constrained_agent], self.goals[constrained_agent], self.heuristics[constrained_agent],
                          constrained_agent, child_node['constraints'])
                if replanned_path is not None:
                    child_node['paths'][constrained_agent] = replanned_path
                    child_node['collisions'] = super().detect_collisions(child_node['paths'])
                    child_node['cost'] = get_sum_of_cost(child_node['paths'])
                    child_node['h'] = self.get_dg_heuristic(self.my_map, child_node['paths'], self.starts, self.goals, self.heuristics, child_node['constraints'])

                    self.push_node(child_node)

        if record_results:
            self.write_results()
            self.print_results(initial_node)
        return initial_node['paths']