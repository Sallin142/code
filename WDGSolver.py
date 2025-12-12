import time as timer
from mdd import buildMDDTree, get_all_optimal_paths, buildJointMDD, check_jointMDD_for_dependency, check_MDDs_for_conflict, balanceMDDs
from mvc import Graph
from single_agent_planner import a_star, get_sum_of_cost
from cbs import CBSSolver, detect_collisions, disjoint_splitting, paths_violate_constraint
import copy
from heuristic_cache import HeuristicCache
from CGSolver import CGSolver
import pulp as pl
from pulp import LpProblem, LpMinimize, LpVariable, value

class WDGSolver(CBSSolver):

    def __init__(self, my_map, starts, goals):
        """Initialize with cache."""
        super().__init__(my_map, starts, goals)
        self.cache = HeuristicCache()

    def get_wdg_heuristic(self, my_map, paths, starts, goals, low_level_h, constraints):
        cached_h = self.cache.get_wdg(constraints)
        if cached_h is not None:
            return int(cached_h)
        
        dependencies = []
        all_paths = []
        all_mdds = []
        
        for i in range(len(paths)):
            cached_mdd = self.cache.get_mdd(i, constraints)
            
            if cached_mdd is not None:
                optimal_paths, root_node, nodes_dict = cached_mdd
            else:
                optimal_paths = get_all_optimal_paths(my_map, starts[i], goals[i], low_level_h[i], i, constraints, max_paths=200)
                if not optimal_paths:
                    return len(paths)  # Upper bound
                
                root_node, nodes_dict = buildMDDTree(optimal_paths)
                self.cache.store_mdd(i, constraints, optimal_paths, root_node, nodes_dict)
            
            all_paths.append(optimal_paths)
            all_mdds.append((root_node, nodes_dict))
        
        # get collisions from solution
        current_collisions = detect_collisions(paths)
        
        # build set of agent pairs that have collisions
        collision_pairs = set()
        for collision in current_collisions:
            a1, a2 = collision['a1'], collision['a2']
            if a1 > a2:
                a1, a2 = a2, a1
            collision_pairs.add((a1, a2))
        
        # check all pairs that have collisions
        for i, j in collision_pairs:
            # check cache first
            cached_dep = self.cache.get_dependency(i, j, constraints)
            
            if cached_dep is not None:
                if cached_dep:
                    dependencies.append((i, j))
                continue
            
            # compute dependency
            paths_i = all_paths[i]
            root_i, nodes_dict_i = all_mdds[i]
            paths_j = all_paths[j]
            root_j, nodes_dict_j = all_mdds[j]
            
            # cardinal conflict check (fast path)
            balanceMDDs(paths_i, paths_j, nodes_dict_i, nodes_dict_j)
            if check_MDDs_for_conflict(nodes_dict_i, nodes_dict_j):
                dependencies.append((i, j))
                self.cache.store_dependency(i, j, constraints, True)
                continue
            
            # joint MDD check
            try:
                joint_root, joint_bottom = buildJointMDD(
                    paths_i, paths_j, 
                    root_i, nodes_dict_i,
                    root_j, nodes_dict_j
                )
                
                is_dependent = check_jointMDD_for_dependency(joint_bottom, paths_i[0], paths_j[0])
                
                if is_dependent:
                    dependencies.append((i, j))
                
                self.cache.store_dependency(i, j, constraints, is_dependent)
                    
            except Exception as e:
                dependencies.append((i, j))
                self.cache.store_dependency(i, j, constraints, True)
        
        if not dependencies:
            self.cache.store_wdg(constraints, 0)
            return 0
        
        # Compute heuristic
        model = LpProblem("WDG_heuristic", LpMinimize)
        lp_agents = {}
        for agent1, agent2 in dependencies:
            if agent1 not in lp_agents:
                lp_agents[agent1] = LpVariable(f"agent{agent1}_weight", lowBound=0, cat="Integer")
            if agent2 not in lp_agents:
                lp_agents[agent2] = LpVariable(f"agent{agent2}_weight", lowBound=0, cat="Integer")

            # # Compute joint paths for agents
            # joint_paths = CGSolver(my_map, [starts[agent1], starts[agent2]], [goals[agent1], goals[agent2]]).find_solution(False)
            # if joint_paths is None:
            #     return -1

            # joint_cost = get_sum_of_cost(joint_paths)
            # individual_cost = len(all_paths[agent1][0]) + len(all_paths[agent2][0])
            # weight = individual_cost - joint_cost

            # # Constraint for edge weight
            # model += lp_agents[agent1] + lp_agents[agent2] >= weight
            # Filter constraints relevant to the pair




            # pair_constraints = [c for c in constraints if c["agent"] in (agent1, agent2)]

            # # Solve joint problem for the two agents
            # joint_solver = CGSolver(my_map,
            #                         [starts[agent1], starts[agent2]],
            #                         [goals[agent1], goals[agent2]])

            # joint_paths = joint_solver.find_solution(disjoint=True,
            #                                         root_constraints=pair_constraints,
            #                                         record_results=False)

            # # If infeasible → upper bound heuristic = len(paths)
            # if joint_paths is None:
            #     weight = 1  # safe minimal positive dependency
            # else:
            #     joint_cost = get_sum_of_cost(joint_paths)
            #     individual_cost = get_sum_of_cost([all_paths[agent1][0],
            #                                     all_paths[agent2][0]])
            #     weight = max(0, individual_cost - joint_cost)

            # # Add LP constraint for this dependent pair
            # model += lp_agents[agent1] + lp_agents[agent2] >= weight


            delta = self.cache.get_pair_weight(agent1, agent2, constraints)

            if delta is None:
                # Compute cost difference via CGSolver
                sub_cg = CGSolver(my_map, [starts[agent1], starts[agent2]],
                                  [goals[agent1], goals[agent2]])
                joint_paths = sub_cg.find_solution(False)

                if joint_paths is None:
                    delta = 1
                else:
                    joint_cost = get_sum_of_cost(joint_paths)
                    cost_i = len(all_paths[agent1][0])
                    cost_j = len(all_paths[agent2][0])
                    delta = (cost_i + cost_j) - joint_cost

                self.cache.store_pair_weight(agent1, agent2, constraints, delta)
            
            model += lp_agents[agent1] + lp_agents[agent2] >= delta




        # Minimize sum of agent weights
        model += sum(lp_agents.values())
        model.solve(pl.PULP_CBC_CMD(msg=False))

        # return value(model.objective)

        # obj = value(model.objective)
        # return int(obj) if obj is not None else len(paths)

        hval = int(value(model.objective))
        self.cache.store_wdg(constraints, hval)
        return hval

    
    def push_node(self, node):
        import heapq
        f_val = node['cost'] + node['h']
        heapq.heappush(self.open_list, (f_val, len(node['collisions']), 
                      self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1
    
    def find_solution(self, disjoint=True, root_constraints=[], root_h=0, record_results=True):
        """Find optimal MAPF solution using CBS with WDG heuristic."""
        self.start_time = timer.time()
   
        root = {
            'cost': 0,
            'h': root_h,
            'constraints': root_constraints,
            'paths': [],
            'collisions': []
        }

        for i in range(self.num_of_agents):
            path = a_star(self.my_map, self.starts[i], self.goals[i], 
                         self.heuristics[i], i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)
        
        root['cost'] = get_sum_of_cost(root['paths'])
        root['h'] = self.get_wdg_heuristic(self.my_map, root['paths'], self.starts, 
                                          self.goals, self.heuristics, root['constraints'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)
        
        while len(self.open_list) > 0:
            curr = self.pop_node()
            
            if not curr['collisions']:
                if record_results:
                    self.print_results(curr)
                    self.cache.print_stats()  # Print cache statistics
                return curr['paths']
            
            collision = curr['collisions'][0]
            constraints = disjoint_splitting(collision) if disjoint else self.standard_splitting(collision)
            
            for constraint in constraints:
                is_conflicting = self._is_conflicting_constraint(constraint, curr['constraints'])
                
                if is_conflicting:
                    continue
                
                child = {}
                child['constraints'] = copy.deepcopy(curr['constraints'])
                if constraint not in child['constraints']:
                    child['constraints'].append(constraint)
                child['paths'] = copy.deepcopy(curr['paths'])
                
                prune_child = False
                if constraint['positive']:
                    conflicted_agents = paths_violate_constraint(constraint, child['paths'])
                    for i in conflicted_agents:
                        new_path = a_star(self.my_map, self.starts[i], self.goals[i], 
                                        self.heuristics[i], i, child['constraints'])
                        if new_path is None:
                            prune_child = True
                            break
                        else:
                            child['paths'][i] = new_path
                
                if prune_child:
                    continue
                
                agent = constraint['agent']
                path = a_star(self.my_map, self.starts[agent], self.goals[agent], 
                            self.heuristics[agent], agent, child['constraints'])
                
                if path is not None:
                    child['paths'][agent] = path
                    child['collisions'] = detect_collisions(child['paths'])
                    child['cost'] = get_sum_of_cost(child['paths'])
                    child['h'] = self.get_wdg_heuristic(self.my_map, child['paths'], 
                                                       self.starts, self.goals, 
                                                       self.heuristics, child['constraints'])
                    
                    self.push_node(child)
        
        if record_results:
            self.print_results(root)
            self.cache.print_stats()
        return None
    
    def _is_conflicting_constraint(self, constraint, existing_constraints):
        """Check if a new constraint conflicts with existing constraints."""
        if constraint in existing_constraints:
            return True
        
        t = constraint['timestep']
        constraints_at_t = [c for c in existing_constraints 
                           if c['timestep'] == t and c['agent'] == constraint['agent']]
        
        is_new_vertex_constraint = len(constraint['loc']) == 1
        
        for old_constraint in constraints_at_t:
            is_old_vertex_constraint = len(old_constraint['loc']) == 1
            
            if is_old_vertex_constraint:
                if old_constraint['positive']:
                    if is_new_vertex_constraint and not constraint['positive'] and \
                       constraint['loc'] == old_constraint['loc']:
                        return True
                    if is_new_vertex_constraint and constraint['positive'] and \
                       constraint['loc'] != old_constraint['loc']:
                        return True
                    if not is_new_vertex_constraint and constraint['positive'] and \
                       constraint['loc'][1] != old_constraint['loc'][0]:
                        return True
                else:
                    if is_new_vertex_constraint and constraint['positive'] and \
                       constraint['loc'] == old_constraint['loc']:
                        return True
                    if not is_new_vertex_constraint and constraint['positive'] and \
                       constraint['loc'][1] == old_constraint['loc'][0]:
                        return True
            else:
                if old_constraint['positive']:
                    if is_new_vertex_constraint and constraint['positive'] and \
                       constraint['loc'][0] != old_constraint['loc'][1]:
                        return True
                    if is_new_vertex_constraint and not constraint['positive'] and \
                       constraint['loc'][0] == old_constraint['loc'][1]:
                        return True
                    if not is_new_vertex_constraint and constraint['positive']:
                        return True
                    if not is_new_vertex_constraint and not constraint['positive'] and \
                       constraint['loc'] == old_constraint['loc']:
                        return True
                else:
                    if not is_new_vertex_constraint and constraint['positive'] and \
                       constraint['loc'] == old_constraint['loc']:
                        return True
        
        return False
    
    def standard_splitting(self, collision):
        """Create standard negative constraints for a collision."""
        constraints = []
        if len(collision['loc']) == 1:
            constraints.append({
                'agent': collision['a1'],
                'loc': collision['loc'],
                'timestep': collision['timestep'],
                'positive': False
            })
            constraints.append({
                'agent': collision['a2'],
                'loc': collision['loc'],
                'timestep': collision['timestep'],
                'positive': False
            })
        else:
            constraints.append({
                'agent': collision['a1'],
                'loc': [collision['loc'][0], collision['loc'][1]],
                'timestep': collision['timestep'],
                'positive': False
            })
            constraints.append({
                'agent': collision['a2'],
                'loc': [collision['loc'][1], collision['loc'][0]],
                'timestep': collision['timestep'],
                'positive': False
            })
        return constraints