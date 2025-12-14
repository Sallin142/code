"""
Memorization: heuristic cache for MAPF solver
reference: 
https://www2.cs.sfu.ca/~hangma/pub/ijcai19.pdf 
6 Runtime Reduction Techniques

MDD construction for individual agents, dependency checking between agent pairs,
computation of pairwise weights delta ij, caching the overall WDG heuristic value per CT node.
"""

# cache for expensive heuristic computations in CBS
class HeuristicCache:

    def __init__(self):
        self.mdd_cache = {}
        self.dependency_cache = {}
        self.delta_cache = {}
        self.wdg_cache = {}
    
    # creates hash key for specified agent constraints 
    def _hash_constraints(self, agent_id, constraints):
        relevant = []
        # filter constraints for specified agent
        for c in constraints:
            if c['agent'] == agent_id:
                # convert location to hashable tuple
                loc_tuple = tuple(c['loc']) if isinstance(c[ 'loc'] , list) else (c['loc'],)
                
                # create constraint tuple
                constraint_tuple = (c['agent'], loc_tuple, c['timestep'], c.get('positive', False))
                relevant.append(constraint_tuple)
        
        # sort in constraint order
        relevant.sort()
        return tuple(relevant)
    
    # create hash key for glabal (entire constarint set)
    def _hash_constraints_global(self, constraints ):
        items = []
        # convert all constraints to hashable tuples
        for c in constraints:
            loc_tuple = tuple(c['loc']) if isinstance(c['loc'], list) else (c['loc'],)
            items.append(( c['agent'], loc_tuple, c['timestep'], bool(c.get( 'positive', False))))
        
        items.sort()
        return tuple(items)
    
    # MDD cache
    def get_mdd(self, agent_id, constraints):
        key = (agent_id, self._hash_constraints(agent_id, constraints))
        return self.mdd_cache.get(key)
    
    def store_mdd(self, agent_id, constraints, optimal_paths, root_node , nodes_dict):
        key = (agent_id, self._hash_constraints(agent_id, constraints ))
        self.mdd_cache[key] = (optimal_paths, root_node, nodes_dict)
    
    # dependency cache
    # get cached dependency for agent pair
    def get_dependency(self, agent_i, agent_j, constraints):
        # make sure i < j
        if agent_i > agent_j:
            agent_i, agent_j = agent_j, agent_i
        
        # create cache key
        hash_i = self._hash_constraints(agent_i, constraints)
        hash_j = self._hash_constraints(agent_j, constraints)
        key = (agent_i, agent_j, hash_i, hash_j )
        
        return self.dependency_cache.get(key)
    
    # store dependency for agents pair
    def store_dependency(self, agent_i, agent_j, constraints, is_dependent):

        # make sure i< j
        if agent_i > agent_j:
            agent_i, agent_j = agent_j, agent_i
        
        # create cache key
        hash_i = self._hash_constraints(agent_i, constraints)
        hash_j = self._hash_constraints(agent_j, constraints)
        key = (agent_i, agent_j, hash_i, hash_j )
        
        self.dependency_cache[key] = is_dependent
    
    #delta cache
    
    # retrieve cached delta value for agent pair
    def get_pair_weight(self, agent_i, agent_j, constraints):
        
        if agent_i > agent_j:
            agent_i, agent_j = agent_j, agent_i
        
        # create cache key
        hash_i = self._hash_constraints(agent_i, constraints)
        hash_j = self._hash_constraints(agent_j, constraints)
        key = (agent_i, agent_j, hash_i, hash_j )
        
        return self.delta_cache.get(key)
    
    # store the delta value
    def store_pair_weight(self, agent_i, agent_j, constraints, delta):

        if agent_i > agent_j:
            agent_i, agent_j = agent_j, agent_i
        
        # create cache key
        hash_i = self._hash_constraints(agent_i, constraints)
        hash_j = self._hash_constraints(agent_j, constraints)
        key = (agent_i, agent_j, hash_i, hash_j )
        
        self.delta_cache[key] = delta

    # WDG heuristic cache
    def get_wdg(self, constraints):
        key = self._hash_constraints_global(constraints)
        return self.wdg_cache.get(key)
    
    # store WDG heuristic value for constraint set
    def store_wdg(self, constraints, heuristic_value):
        key = self._hash_constraints_global(constraints )
        self.wdg_cache[key] = int( heuristic_value)
    
    def clear(self):
        self.mdd_cache.clear()
        self.dependency_cache.clear()
        self.delta_cache.clear()
        self.wdg_cache.clear()