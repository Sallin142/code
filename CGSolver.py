"""
Conflict-Based Search Heuristic
"""

import time as timer
import copy
from mdd import buildMDDTree, get_all_optimal_paths, check_MDDs_for_conflict, balanceMDDs
from mvc import Graph
from single_agent_planner import a_star, get_sum_of_cost
from cbs import CBSSolver, detect_collisions, disjoint_splitting, paths_violate_constraint, standard_splitting


class CGSolver(CBSSolver):
 
    def get_cg_heuristic(self, my_map, paths, starts, goals, low_level_h, constraints, all_paths = None, all_mdds = None, max_paths = 200):
        """
        Compute CG heuristic using cardinal conflicts
        
        Algorithm:
        - Build MDDs for all agents
        - Check for cardinal conflicts
        - Build conflict graph from cardinal conflicts
        - Compute minimum vertex cover size as h-value

        """
        # exit if the constraints are too large
        if len(constraints ) > 5:
            return 1
        
        cardinal_conflicts = []
        
        # initialize caches
        if all_paths is None:
            all_paths = []
        if all_mdds is None:
            all_mdds = []

        # build MDDs for all agents
        if len(all_paths ) == 0 and len(all_mdds ) == 0:
            for i in range(len(paths)):
                # set all optimal paths
                optimal_paths = get_all_optimal_paths( my_map, starts[i], goals[i], low_level_h[i], i, constraints, max_paths = max_paths)
                
                if not optimal_paths:
                    return -1
                
                # build MDD from optimal paths
                _, nodes_dict = buildMDDTree(optimal_paths )
                all_paths.append(optimal_paths)
                all_mdds.append(nodes_dict)

        # check if all agent pair for cardinal conflicts
        for i in range(len(paths )):
            paths_i = all_paths[i]
            mdd_i = all_mdds[i]

            for j in range(i + 1, len(paths)):
                paths_j = all_paths[j]
                mdd_j = all_mdds[j]
                
                # balance MDDs to same length
                balanceMDDs(paths_i, paths_j, mdd_i, mdd_j)
                if check_MDDs_for_conflict(mdd_i, mdd_j):
                    cardinal_conflicts.append((i, j))
        
        # build conflict graph from cardinal conflicts
        graph = Graph(len( paths))
        for i, j in cardinal_conflicts:
            graph.addEdge(i, j)
        
        # compute minimum vertex cover
        vertex_cover = graph.getVertexCover()
        return len(vertex_cover )

    # find the solution using cbs with cg
    def find_solution(self, disjoint = True, root_constraints = [], root_h = 0, record_results = True):
        self.start_time = timer.time()

        root = {
            'cost': 0,
            'h': root_h,
            'constraints': root_constraints,
            'paths': [],
            'collisions': []
        }
        
        # compute initial paths for all agents
        for i in range(self.num_of_agents):
            path = a_star( self.my_map, self.starts[i], self.goals[i] , 
                self.heuristics[i] , i, root['constraints'])
            if path is None:
                raise BaseException('No solutions' )
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['h'] = self.get_cg_heuristic( self.my_map, root['paths'], self.starts, self.goals, 
                                          self.heuristics, root['constraints'])
        # for real_map_benchmark
        self.root_h_value = root['h']
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        # A* search with cg
        while len(self.open_list) > 0:
            curr = self.pop_node()

            if not curr['collisions']:
                if record_results:
                    self.print_results(curr )
                return curr['paths']
            
            # split on first collision
            collision = curr['collisions'][0]
            constraints = disjoint_splitting(collision) if disjoint else standard_splitting( collision)
            
            # generate child nodes
            for constraint in constraints:
                # check if constraint conflicts with existing constraints
                if self._is_conflicting_constraint( constraint, curr['constraints']):
                    continue
                
                child = {}
                child['constraints'] = copy.deepcopy(curr['constraints'] )
                if constraint not in child['constraints']:
                    child['constraints'].append( constraint )
                child['paths'] = copy.deepcopy(curr['paths'])

                # disjoint splitting to handle the positive constraints
                prune_child = False
                if constraint['positive']:
                    # replan for agents that violate the positive constraint
                    conflicted_agents = paths_violate_constraint( constraint, child['paths'])
                    for i in conflicted_agents:
                        new_path = a_star( 
                            self.my_map, self.starts[i], self.goals[i], 
                            self.heuristics[i], i, child['constraints']
                        )
                        if new_path is None:
                            prune_child = True
                            break
                        else:
                            child['paths'][i] = new_path
                
                if prune_child:
                    continue

                # replan for the constrained agent
                agent = constraint['agent']
                path = a_star(
                    self.my_map, self.starts[agent] , self.goals[agent], 
                    self.heuristics[agent] , agent, child['constraints']
                )
                
                # add child to open list if valid path found
                if path is not None:
                    child['paths'][agent] = path
                    child['collisions'] = detect_collisions( child['paths'])
                    child['cost'] = get_sum_of_cost(child['paths'])
                    child['h'] = self.get_cg_heuristic(
                        self.my_map, child['paths'], 
                        self.starts, self.goals, 
                        self.heuristics, child['constraints' ]
                    )
                    self.push_node(child )

        if record_results:
            self.print_results(root)
        return None
    
    # check if new constraint conflict the existing
    def _is_conflicting_constraint(self, constraint, existing_constraints ):

        # duplicate constraint
        if constraint in existing_constraints:
            return True
        
        # get constraints at same timestep for same agent
        t = constraint['timestep']
        constraints_at_t = [
            c for c in existing_constraints 
            if c['timestep'] == t and c['agent'] ==constraint['agent']
        ]
        
        is_new_vertex = len(constraint['loc']) == 1
        
        # check conflicts with each existing constraint
        for old in constraints_at_t:
            is_old_vertex = len(old['loc']) == 1
            
            # when old constraint is vertex constraint
            if is_old_vertex:
                if old['positive']:
                    # old is positive vertex constraint
                    if is_new_vertex and not constraint['positive'] and constraint['loc'] == old['loc']:
                        return True
                    if is_new_vertex and constraint['positive'] and constraint['loc'] != old['loc']:
                        return True
                    if not is_new_vertex and constraint['positive'] and constraint['loc'][1] != old['loc'][0]:
                        return True
                else:
                    if is_new_vertex and constraint['positive'] and constraint['loc'] == old['loc']:
                        return True
                    if not is_new_vertex and constraint['positive'] and constraint['loc'][1] == old['loc'][0]:
                        return True
     
            #  or when old constraint is edge constraint
            else:
                if old['positive']:
                    if is_new_vertex and constraint['positive'] and constraint['loc'][0] != old['loc'][1]:
                        return True
                    if is_new_vertex and not constraint['positive'] and constraint['loc'][0] == old['loc'][1]:
                        return True
                    if not is_new_vertex and constraint['positive']:
                        return True
                    if not is_new_vertex and not constraint['positive'] and constraint['loc'] == old['loc']:
                        return True
                else:
                    # old is negative edge constraint
                    if not is_new_vertex and constraint['positive'] and constraint['loc'] == old['loc']:
                        return True
        
        return False