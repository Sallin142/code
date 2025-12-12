import time as timer
from mdd import buildMDDTree, get_all_optimal_paths, buildJointMDD, check_jointMDD_for_dependency, check_MDDs_for_conflict, balanceMDDs
from mvc import Graph
from single_agent_planner import a_star, get_sum_of_cost
from cbs import CBSSolver, detect_collisions, disjoint_splitting, paths_violate_constraint
import copy
from CGSolver import CGSolver
import pulp as pl
from pulp import LpProblem, LpMinimize, LpVariable, value
from heuristic_cache import HeuristicCache


class WDGSolver(CBSSolver):


    def __init__(self, my_map, starts, goals):
        super().__init__(my_map, starts, goals)
        self.cache = HeuristicCache()
    
    def get_wdg_heuristic(self, my_map, paths, starts, goals, low_level_h, constraints, debug=False):
        dependencies = []  # List of (agent1, agent2, weight) tuples
        
        all_paths = []
        all_mdds = []
        
        timeout_count = 0  # Track consecutive timeouts
        max_timeouts = 3   # Stop trying after 3 timeouts
        
        if debug:
            print(f"\n[WDG DEBUG] Starting heuristic calculation for {len(paths)} agents")
        
        # Build MDDs for all agents (WITH CACHING!)
        for i in range(len(paths)):
            # Check cache first
            cached_mdd = self.cache.get_mdd(i, constraints)
            
            if cached_mdd is not None:
                optimal_paths, root_node, nodes_dict = cached_mdd
                if debug:
                    print(f"[WDG DEBUG] Agent {i}: Using cached MDD, {len(optimal_paths)} paths, cost={len(optimal_paths[0])-1}")
            else:
                # Cache miss - compute MDD
                # OPTIMIZATION: Limit max_paths to avoid exponential explosion
                optimal_paths = get_all_optimal_paths(my_map, starts[i], goals[i], low_level_h[i], i, constraints, max_paths=200)
                if not optimal_paths:
                    if debug:
                        print(f"[WDG DEBUG] Agent {i}: No optimal paths found!")
                    return 0  # Return 0 instead of -1
                
                root_node, nodes_dict = buildMDDTree(optimal_paths)
                
                # Store in cache
                self.cache.store_mdd(i, constraints, optimal_paths, root_node, nodes_dict)
                
                if debug:
                    print(f"[WDG DEBUG] Agent {i}: Computed MDD, {len(optimal_paths)} paths, cost={len(optimal_paths[0])-1}")
            
            all_paths.append(optimal_paths)
            all_mdds.append((root_node, nodes_dict))
        
        # Get collisions from CURRENT solution
        current_collisions = detect_collisions(paths)
        
        if debug:
            print(f"[WDG DEBUG] Found {len(current_collisions)} collisions in current solution")
        
        # Build set of agent pairs that have collisions
        collision_pairs = set()
        for collision in current_collisions:
            a1, a2 = collision['a1'], collision['a2']
            if a1 > a2:
                a1, a2 = a2, a1
            collision_pairs.add((a1, a2))
        
        # Check all pairs that have collisions
        for i, j in collision_pairs:
            if debug:
                print(f"[WDG DEBUG] Checking agents {i} and {j}")
            
            paths_i = all_paths[i]
            root_i, nodes_dict_i = all_mdds[i]
            paths_j = all_paths[j]
            root_j, nodes_dict_j = all_mdds[j]
            
            # Check cache for dependency first
            cached_dep = self.cache.get_dependency(i, j, constraints)
            
            if cached_dep is not None:
                is_dependent = cached_dep
                if debug:
                    print(f"[WDG DEBUG] Agents {i} and {j}: Using cached dependency = {is_dependent}")
            else:
                # Compute dependency
                balanceMDDs(paths_i, paths_j, nodes_dict_i, nodes_dict_j)
                is_cardinal = check_MDDs_for_conflict(nodes_dict_i, nodes_dict_j)
                
                if is_cardinal:
                    is_dependent = True
                    if debug:
                        print(f"[WDG DEBUG] Agents {i} and {j}: CARDINAL conflict")
                else:
                    # Build joint MDD
                    try:
                        joint_root, joint_bottom = buildJointMDD(
                            paths_i, paths_j, 
                            root_i, nodes_dict_i,
                            root_j, nodes_dict_j
                        )
                        is_dependent = check_jointMDD_for_dependency(joint_bottom, paths_i[0], paths_j[0])
                    except Exception as e:
                        is_dependent = True
                        if debug:
                            print(f"[WDG DEBUG] Agents {i} and {j}: Exception: {e}")
                
                # Store in cache
                self.cache.store_dependency(i, j, constraints, is_dependent)
            
            if not is_dependent:
                if debug:
                    print(f"[WDG DEBUG] Agents {i} and {j}: INDEPENDENT")
                continue
            
            # Compute Δij (WITH CACHING!)
            cached_delta = self.cache.get_delta(i, j, constraints)
            
            if cached_delta is not None:
                weight = cached_delta
                if debug:
                    print(f"[WDG DEBUG] Agents {i} and {j}: Using cached Δij = {weight}")
            else:
                # Early stopping: if too many timeouts, use fallback
                if timeout_count >= max_timeouts:
                    if debug:
                        print(f"[WDG DEBUG] Agents {i} and {j}: Too many timeouts, using Δij=1 (fast fallback)")
                    weight = 1
                    self.cache.store_delta(i, j, constraints, weight)
                    dependencies.append((i, j, weight))
                    continue
                
                # Compute delta
                joint_constraints = [c for c in constraints 
                                   if c['agent'] == i or c['agent'] == j]
                
                # Map agent indices: i->0, j->1
                two_agent_constraints = []
                for c in joint_constraints:
                    new_c = copy.deepcopy(c)
                    if c['agent'] == i:
                        new_c['agent'] = 0
                    elif c['agent'] == j:
                        new_c['agent'] = 1
                    two_agent_constraints.append(new_c)
                
                if debug:
                    print(f"[WDG DEBUG] Agents {i} and {j}: Computing Δij with {len(two_agent_constraints)} constraints")
                
                try:
                    joint_solver = CGSolver(my_map, [starts[i], starts[j]], 
                                          [goals[i], goals[j]])
                    
                    # Use tighter timeout for 2-agent problems
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("2-agent CBS timeout")
                    
                    # Set alarm for 0.3 seconds (aggressive timeout)
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(0)  # 0 = use fractional seconds
                    signal.setitimer(signal.ITIMER_REAL, 0.3)  # 0.3 second timeout
                    
                    try:
                        joint_paths = joint_solver.find_solution(
                            disjoint=False, 
                            root_constraints=two_agent_constraints,
                            root_h=1,  # Paper's optimization
                            record_results=False
                        )
                        signal.setitimer(signal.ITIMER_REAL, 0)  # Cancel timer
                        
                        if joint_paths is None:
                            if debug:
                                print(f"[WDG DEBUG] Failed to find joint paths for {i},{j}")
                            weight = 1  # Conservative fallback
                        else:
                            joint_cost = get_sum_of_cost(joint_paths)
                            individual_cost = (len(paths[i]) - 1) + (len(paths[j]) - 1)
                            weight = max(1, joint_cost - individual_cost)
                            
                            if debug:
                                print(f"[WDG DEBUG] Agents {i}-{j}: individual={individual_cost}, joint={joint_cost}, Δij={weight}")
                    
                    except TimeoutError:
                        signal.setitimer(signal.ITIMER_REAL, 0)  # Cancel timer
                        timeout_count += 1  # Track timeouts
                        if debug:
                            print(f"[WDG DEBUG] Agents {i},{j}: Timeout ({timeout_count}/{max_timeouts}), using Δij=1")
                        weight = 1
                
                except Exception as e:
                    if debug:
                        print(f"[WDG DEBUG] Exception solving two-agent for {i},{j}: {e}")
                    weight = 1
                
                # Store in cache
                self.cache.store_delta(i, j, constraints, weight)
            
            dependencies.append((i, j, weight))
        
        if debug:
            print(f"[WDG DEBUG] Total dependencies: {len(dependencies)}")
        
        # If no dependencies, h-value is 0
        if not dependencies:
            if debug:
                print(f"[WDG DEBUG] No dependencies, returning 0")
            return 0

        # Check if all weights are 1
        weights = [w for _, _, w in dependencies]
        if all(w == 1 for w in weights):
            if debug:
                print(f"[WDG DEBUG] All weights are 1, using unweighted MVC")
            g = Graph(len(paths))
            for agent1, agent2, _ in dependencies:
                g.addEdge(agent1, agent2)
            vertex_cover = g.getVertexCover()
            return len(vertex_cover)
        
        # Solve EWMVC using LP
        try:
            model = LpProblem("WDG_heuristic", LpMinimize)
            lp_agents = {}
            
            agents_in_deps = set()
            for agent1, agent2, weight in dependencies:
                agents_in_deps.add(agent1)
                agents_in_deps.add(agent2)
            
            for agent in agents_in_deps:
                lp_agents[agent] = LpVariable(f"agent{agent}_weight", lowBound=0, cat="Integer")

            for agent1, agent2, weight in dependencies:
                model += lp_agents[agent1] + lp_agents[agent2] >= weight
            
            model += sum(lp_agents.values())
            
            model.solve(pl.PULP_CBC_CMD(msg=False))
            
            result = value(model.objective)
            
            if debug:
                print(f"[WDG DEBUG] LP solver result: {result}")
            
            if result is None:
                if debug:
                    print(f"[WDG DEBUG] LP failed, falling back to MVC")
                g = Graph(len(paths))
                for agent1, agent2, _ in dependencies:
                    g.addEdge(agent1, agent2)
                vertex_cover = g.getVertexCover()
                return len(vertex_cover)
            
            return int(round(result))
        
        except Exception as e:
            if debug:
                print(f"[WDG DEBUG] LP exception: {e}")
            return 0
    
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
    
    def push_node(self, node):
        """Push node to open list with f = g + h prioritization."""
        import heapq
        f_val = node['cost'] + node['h']
        heapq.heappush(self.open_list, (f_val, len(node['collisions']), 
                                       self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1
    
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