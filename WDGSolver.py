import time as timer
from mdd import check_jointMDD_for_dependency, buildJointMDD, buildMDDTree, get_all_optimal_paths
from single_agent_planner import a_star, get_sum_of_cost
from cbs import CBSSolver
import copy
from CGSolver import CGSolver
import pulp as pl
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize, value

class WDGSolver(CBSSolver):
    def get_wdg_heuristic(self, my_map, paths, starts, goals, low_level_h, constraints):
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

        lp_variables_by_agent = {}

        lp_model = LpProblem("edge_weighted_minimum_vertex_cover", LpMinimize)
        for agent_pair in agent_pair_dependencies:
            first_agent_idx = agent_pair[0]
            second_agent_idx = agent_pair[1]
            if first_agent_idx not in lp_variables_by_agent:
                first_agent_var = LpVariable('a'+str(first_agent_idx), lowBound=0, cat="Integer", e=None)
                lp_variables_by_agent[first_agent_idx] = first_agent_var
            if second_agent_idx not in lp_variables_by_agent:
                
                second_agent_var = LpVariable('a'+str(second_agent_idx), lowBound=0, cat="Integer", e=None)
                lp_variables_by_agent[second_agent_idx] = second_agent_var

            pairwise_starts = [starts[first_agent_idx], starts[second_agent_idx]]
            pairwise_goals = [goals[first_agent_idx], goals[second_agent_idx]]
            
            pairwise_solver = CGSolver(my_map, pairwise_starts, pairwise_goals)
            joint_solution_paths = pairwise_solver.find_solution(record_results = False)

            if (joint_solution_paths == []):
                return -1
            joint_optimal_cost = get_sum_of_cost(joint_solution_paths)

            individual_path_lengths = len(optimal_path_sets[first_agent_idx][0]) + len(optimal_path_sets[second_agent_idx][0])
            conflict_cost = individual_path_lengths - joint_optimal_cost
            
            lp_model += lp_variables_by_agent[first_agent_idx] + lp_variables_by_agent[second_agent_idx] >= conflict_cost
        
        objective_function = None
        for agent_variable in lp_variables_by_agent.values():
            objective_function += agent_variable

        lp_model += objective_function

        lp_model.solve(pl.PULP_CBC_CMD(msg=False))

        heuristic_value = value(lp_model.objective)
        return heuristic_value

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
        initial_node['h'] = self.get_wdg_heuristic(self.my_map, initial_node['paths'], self.starts, self.goals, self.heuristics, initial_node['constraints']) 
        if (initial_node['h'] == -1):
            raise BaseException('No solution')
    
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
                    child_node['h'] = self.get_wdg_heuristic(self.my_map, child_node['paths'], self.starts, self.goals, self.heuristics, child_node['constraints'])
                    
                    if (child_node['h'] != -1):
                        self.push_node(child_node)

        if record_results:
            self.print_results(initial_node)
            self.write_results()
        return initial_node['paths']