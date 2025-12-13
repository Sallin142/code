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
        dependencies = []

        all_roots = []
        all_paths = []
        all_mdds = []

        for i in range(len(paths)):
            path = get_all_optimal_paths(my_map, starts[i], goals[i], low_level_h[i], i, constraints)
            root, nodes_dict = buildMDDTree(path)
            all_roots.append(root)
            all_paths.append(path)
            all_mdds.append(nodes_dict)

        for i in range(len(paths)): # num of agents in map
            paths1 = all_paths[i]
            root1 = all_roots[i]
            node_dict1 = all_mdds[i]

            for j in range(i+1,len(paths)):
                paths2 = all_paths[j]
                root2 = all_roots[j]
                node_dict2 = all_mdds[j]
                
                root, bottom_node = buildJointMDD(paths1, paths2, root1, node_dict1, root2, node_dict2)

                if (check_jointMDD_for_dependency(bottom_node, paths1, paths2)):
                    dependencies.append((i,j))

        dependent_agents_dict = {}

        model = LpProblem("edge_weighted_minimum_vertex_cover", LpMinimize)
        for dependency in dependencies:
            agent1 = dependency[0]
            agent2 = dependency[1]
            if agent1 not in dependent_agents_dict:
                a1 = LpVariable('a'+str(agent1), lowBound=0, cat="Integer", e=None)
                dependent_agents_dict[agent1] = a1
            if agent2 not in dependent_agents_dict:
                # add lp var 
                a2 = LpVariable('a'+str(agent2), lowBound=0, cat="Integer", e=None)
                dependent_agents_dict[agent2] = a2

            new_starts = [starts[agent1], starts[agent2]]
            new_goals = [goals[agent1], goals[agent2]]
            
            cgs = CGSolver(my_map, new_starts, new_goals)
            paths = cgs.find_solution(record_results = False)

            if (paths == []):
                return -1
            min_cost = get_sum_of_cost(paths)

            sum_indv_opt_paths = len(all_paths[agent1][0]) + len(all_paths[agent2][0])
            edge_weight = sum_indv_opt_paths - min_cost
            # add LP constraints for the edge
            model += dependent_agents_dict[agent1] + dependent_agents_dict[agent2] >= edge_weight
        
        objective = None
        for lp_var in dependent_agents_dict.values():
            objective += lp_var

        model += objective

        model.solve(pl.PULP_CBC_CMD(msg=False))

        h = value(model.objective)
        return h

    def find_solution(self, disjoint=True, record_results=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        root = {'cost': 0,
                'h': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['h'] = self.get_wdg_heuristic(self.my_map, root['paths'], self.starts, self.goals, self.heuristics, root['constraints']) 
        if (root['h'] == -1):
            raise BaseException('No solution')
    
        root['collisions'] = super().detect_collisions(root['paths'])
        self.push_node(root)

        while len(self.open_list) > 0:
            curr = self.pop_node()

            if not curr['collisions']:
                if record_results:
                    self.print_results(curr)
                    self.write_results()
                return curr['paths'] # this is the goal node
            
            collision = curr['collisions'][0]
            # constraints = standard_splitting(collision)
            constraints = super().disjoint_splitting(collision)
            for constraint in constraints:
                if super().is_conflicting_constraint(constraint, curr['constraints']):
                    continue
                child = {}
                child['constraints'] = copy.deepcopy(curr['constraints'])
                if constraint not in child['constraints']:
                    child['constraints'].append(constraint)
                child['paths']= copy.deepcopy(curr['paths'])

                prune_child = False
                if constraint['positive']:
                    conflicted_agents = super().paths_violate_constraint(constraint, child['paths'])
                    for i in conflicted_agents:
                        new_path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                            i, child['constraints'])
                        if new_path is None:
                            prune_child = True
                            break
                        else:
                            child['paths'][i] = new_path
                if prune_child:
                    continue

                agent = constraint['agent']
                path = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent],
                          agent, child['constraints'])
                if path is not None:
                    child['paths'][agent] = path
                    child['collisions'] = super().detect_collisions(child['paths'])
                    child['cost'] = get_sum_of_cost(child['paths'])
                    child['h'] = self.get_wdg_heuristic(self.my_map, child['paths'], self.starts, self.goals, self.heuristics, child['constraints'])
                    # child['h'] = get_dg_heuristic(self.my_map, child['paths'], self.starts, self.goals, self.heuristics, child['constraints'])
                    if (child['h'] != -1):
                        self.push_node(child)

        if record_results:
            self.print_results(root)
            self.write_results()
        return root['paths']