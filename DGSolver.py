"""
DGSolver improves the CG heuristic

Reference:  Li, J., Felner, A., Boyarski, E., Ma, H., & Koenig, S. (2019). "Improved Heuristics for Multi-Agent Path 
Finding with Conflict-Based Search."

"""
import time as timer
import copy
from mdd import (buildMDDTree, get_all_optimal_paths, buildJointMDD, check_jointMDD_for_dependency, check_MDDs_for_conflict, balanceMDDs)
from mvc import Graph
from single_agent_planner import a_star, get_sum_of_cost
from cbs import CBSSolver, detect_collisions, disjoint_splitting, paths_violate_constraint, standard_splitting
from heuristic_cache import HeuristicCache


class DGSolver(CBSSolver):

    def __init__(self, my_map, starts, goals):
        """
        initialize the  dgsolver with cache for memorization
        """
        super().__init__(my_map, starts, goals)
        self.cache = HeuristicCache()

    def get_dg_heuristic(self, my_map, paths, starts, goals, low_level_h, constraints):
        """
        
        Algorithm:
        - Build MDDs for all agents
        - For each colliding pair:
           if path is fast: check for cardinal conflict
           if path is slow: build joint MDD and check dependency
        - Build dependency graph from dependent pairs
        - Compute minimum vertex cover size as h-value
        """
        # exit if the constraints are too large
        if len(constraints) > 5:
            return 1
        
        dependencies = []
        all_paths = []
        # store MDDs for each agent
        all_mdds = []
        
        # build mdd for all agents and get cached mdd
        for i in range(len( paths)):
            cached_mdd = self.cache.get_mdd(i, constraints )
            
            if cached_mdd is not None:
                # use cached mdd
                optimal_paths, root_node, nodes_dict = cached_mdd
            else:
                # compute new mdd
                optimal_paths = get_all_optimal_paths(
                    my_map, starts[i], goals[i], low_level_h[i], i, constraints, max_paths = 200
                )

                if not optimal_paths:
                    return len(paths)
                
                root_node, nodes_dict = buildMDDTree(optimal_paths)             
                # store in cache
                self.cache.store_mdd(i, constraints, optimal_paths, root_node, nodes_dict)
            
            all_paths.append(optimal_paths)
            all_mdds.append((root_node , nodes_dict))
        
        current_collisions = detect_collisions(paths)
        
        # build set of agent pairs that have collisions
        collision_pairs = set()
        for collision in current_collisions:
            a1, a2 = collision['a1'], collision['a2' ]
            if a1 > a2:
                a1, a2 = a2, a1
            collision_pairs.add(( a1, a2))
        
        # check dependency for each colliding pair
        for i, j in collision_pairs:
            # try to get cached dependency result
            cached_dependency = self.cache.get_dependency(i, j, constraints )
            
            if cached_dependency is not None:
                if cached_dependency:
                    dependencies.append((i, j))
                continue
            
            # compute dependency
            paths_i = all_paths[i]
            root_i, nodes_dict_i = all_mdds[i]
            paths_j = all_paths[j]
            root_j, nodes_dict_j = all_mdds[j]
            
            # for fast path:  cardinal conflict check
            balanceMDDs(paths_i, paths_j, nodes_dict_i, nodes_dict_j)
            if check_MDDs_for_conflict(nodes_dict_i, nodes_dict_j):
                dependencies.append((i, j))
                self.cache.store_dependency(i, j, constraints, True )
                continue
            
            # for slow path joint MDD check
            try:
                # build joint MDD for agent pair
                joint_root, joint_bottom =buildJointMDD(
                    paths_i, paths_j, 
                    root_i, nodes_dict_i,
                    root_j, nodes_dict_j
                )
                
                # check if agents are dependent
                is_dependent = check_jointMDD_for_dependency(joint_bottom, paths_i[0], paths_j[0] )
                
                if is_dependent:
                    dependencies.append((i, j))
                
                # cache the result
                self.cache.store_dependency(i, j, constraints, is_dependent )
                    
            except Exception:
                dependencies.append((i, j))
                self.cache.store_dependency(i, j, constraints, True)
        
        # build dependency graph and compute mvc
        graph = Graph(len(paths))
        for i, j in dependencies:
            graph.addEdge(i, j)
        
        vertex_cover = graph.getVertexCover()
        return len(vertex_cover)
    
    # find optimal solution using cbs with dg h
    def find_solution(self, disjoint = True, root_constraints = [], root_h = 0, record_results = True):
        self.start_time = timer.time()
   
        # initialize root node
        root = {
            'cost': 0,
            'h': root_h,
            'constraints': root_constraints,
            'paths': [],
            'collisions': []
        }

        # compute initial paths for all agents
        for i in range(self.num_of_agents):
            path = a_star(
                self.my_map, self.starts[i], self.goals[i], 
                self.heuristics[i], i, root[ 'constraints']
            )
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)
        
        # compute root node properties
        root['cost'] = get_sum_of_cost(root['paths'])
        root['h'] = self.get_dg_heuristic(
            self.my_map, root['paths'] , self.starts, 
            self.goals, self.heuristics, root['constraints' ]
        )
        self.root_h_value = root['h']
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)
        
        # a star search with dg heuristic
        while len(self.open_list) > 0:
            curr = self.pop_node()
            if not curr['collisions']:
                if record_results:
                    self.print_results(curr)
                return curr['paths']
            
            # split on first collision
            collision = curr['collisions'][0]
            constraints = disjoint_splitting(collision ) if disjoint else standard_splitting(collision )
            
            # generate child nodes
            for constraint in constraints:
                # check if constraint conflicts with existing constraints
                if self._is_conflicting_constraint(constraint, curr['constraints']):
                    continue
                child = {}
                child['constraints'] = copy.deepcopy(curr['constraints'])
                if constraint not in child['constraints']:
                    child['constraints'].append(constraint)
                child['paths'] = copy.deepcopy(curr['paths'])
                
                # disjoint splitting
                prune_child = False
                if constraint['positive']:
                    # replan for agents that violate the positive constraint
                    conflicted_agents = paths_violate_constraint(constraint, child['paths'])
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
                    self.my_map, self.starts[agent], self.goals[agent], 
                    self.heuristics[agent], agent, child['constraints']
                )
                
                # add child to open list
                if path is not None:
                    child['paths'][agent] = path
                    child['collisions'] = detect_collisions(child['paths'])
                    child['cost'] = get_sum_of_cost(child[ 'paths'])
                    child['h'] = self.get_dg_heuristic(
                        self.my_map, child['paths'], 
                        self.starts, self.goals, 
                        self.heuristics, child['constraints']
                    )
                    self.push_node(child)
        
        if record_results:
            self.print_results(root)
        return None
    
    def _is_conflicting_constraint(self, constraint, existing_constraints):
        if constraint in existing_constraints:
            return True
        
        # get constraints at same timestep for same agent
        t = constraint['timestep']
        constraints_at_t = [
            c for c in existing_constraints 
            if c['timestep' ] == t and c['agent'] == constraint['agent']
        ]
        
        is_new_vertex = len(constraint['loc']) == 1
        
        # check conflicts with each existing constraint
        for old in constraints_at_t:
            is_old_vertex = len(old['loc']) == 1
            
            if is_old_vertex:
                if old['positive']:
                    if is_new_vertex and not constraint['positive'] and constraint['loc'] == old['loc']:
                        return True
                    if is_new_vertex and constraint['positive'] and constraint['loc'] != old['loc']:
                        return True
                    if not is_new_vertex and constraint['positive'] and constraint['loc'][1] != old['loc'][0]:
                        return True
                else:
                    # old is negative vertex constraint
                    if is_new_vertex and constraint['positive'] and constraint['loc'] == old['loc']:
                        return True
                    if not is_new_vertex and constraint['positive'] and constraint['loc'][1] == old['loc'][0]:
                        return True
            
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
                    if not is_new_vertex and constraint['positive'] and constraint['loc'] == old['loc']:
                        return True
        
        return False
