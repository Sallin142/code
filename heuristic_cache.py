class HeuristicCache:

    def __init__(self):
        # MDD cache: (agent_id, constraints_hash) -> (optimal_paths, root_node, nodes_dict)
        self.mdd_cache = {}
        
        # Dependency cache: (agent_i, agent_j, constraints_hash) -> bool
        self.dependency_cache = {}
        
        # Delta cache: (agent_i, agent_j, constraints_hash) -> int
        self.delta_cache = {}

        # WDG heuristic cache: constraints_key -> int
        self.wdg_cache = {}
        
        # Statistics
        self.mdd_hits = 0
        self.mdd_misses = 0
        self.dep_hits = 0
        self.dep_misses = 0
        self.delta_hits = 0
        self.delta_misses = 0
        self.wdg_hits = 0
        self.wdg_misses = 0
    
    def _hash_constraints(self, agent_id, constraints):

        relevant = []
        for c in constraints:
            if c['agent'] == agent_id:
                # Create hashable representation
                loc_tuple = tuple(c['loc']) if isinstance(c['loc'], list) else (c['loc'],)
                constraint_tuple = (
                    c['agent'],
                    loc_tuple,
                    c['timestep'],
                    c.get('positive', False)
                )
                relevant.append(constraint_tuple)
        
        # Sort for consistency
        relevant.sort()
        return tuple(relevant)
    
    def _hash_constraints_global(self, constraints):
        """
        Produce an order-independent global signature for the full constraint set.
        Each constraint represented as (agent, loc-tuple, timestep, positive).
        Sorted to make it independent of ordering.
        """
        items = []
        for c in constraints:
            loc_tuple = tuple(c['loc']) if isinstance(c['loc'], list) else (c['loc'],)
            items.append((c['agent'], loc_tuple, c['timestep'], bool(c.get('positive', False))))
        items.sort()
        return tuple(items)
    
    def get_mdd(self, agent_id, constraints):

        key = (agent_id, self._hash_constraints(agent_id, constraints))
        
        if key in self.mdd_cache:
            self.mdd_hits += 1
            return self.mdd_cache[key]
        else:
            self.mdd_misses += 1
            return None
    
    def store_mdd(self, agent_id, constraints, optimal_paths, root_node, nodes_dict):
        """Store MDD in cache."""
        key = (agent_id, self._hash_constraints(agent_id, constraints))
        self.mdd_cache[key] = (optimal_paths, root_node, nodes_dict)
    
    def get_dependency(self, agent_i, agent_j, constraints):

        # Ensure i < j for consistency
        if agent_i > agent_j:
            agent_i, agent_j = agent_j, agent_i
        
        hash_i = self._hash_constraints(agent_i, constraints)
        hash_j = self._hash_constraints(agent_j, constraints)
        key = (agent_i, agent_j, hash_i, hash_j)
        
        if key in self.dependency_cache:
            self.dep_hits += 1
            return self.dependency_cache[key]
        else:
            self.dep_misses += 1
            return None
    
    def store_dependency(self, agent_i, agent_j, constraints, is_dependent):
        # Ensure i < j for consistency
        if agent_i > agent_j:
            agent_i, agent_j = agent_j, agent_i
        
        hash_i = self._hash_constraints(agent_i, constraints)
        hash_j = self._hash_constraints(agent_j, constraints)
        key = (agent_i, agent_j, hash_i, hash_j)
        
        self.dependency_cache[key] = is_dependent
    
    def get_delta(self, agent_i, agent_j, constraints):
        # Ensure i < j for consistency
        if agent_i > agent_j:
            agent_i, agent_j = agent_j, agent_i
        
        hash_i = self._hash_constraints(agent_i, constraints)
        hash_j = self._hash_constraints(agent_j, constraints)
        key = (agent_i, agent_j, hash_i, hash_j)
        
        if key in self.delta_cache:
            self.delta_hits += 1
            return self.delta_cache[key]
        else:
            self.delta_misses += 1
            return None
    
    def store_delta(self, agent_i, agent_j, constraints, delta):
        """Store delta value in cache."""
        # Ensure i < j for consistency
        if agent_i > agent_j:
            agent_i, agent_j = agent_j, agent_i
        
        hash_i = self._hash_constraints(agent_i, constraints)
        hash_j = self._hash_constraints(agent_j, constraints)
        key = (agent_i, agent_j, hash_i, hash_j)
        
        self.delta_cache[key] = delta

    def get_pair_weight(self, agent_i, agent_j, constraints):
        """Wrapper for get_delta (more descriptive)."""
        return self.get_delta(agent_i, agent_j, constraints)

    def store_pair_weight(self, agent_i, agent_j, constraints, delta):
        """Wrapper for store_delta (more descriptive)."""
        self.store_delta(agent_i, agent_j, constraints, delta)

    def get_wdg(self, constraints):
        """Return cached WDG heuristic value for the full constraint set."""
        key = self._hash_constraints_global(constraints)
        if key in self.wdg_cache:
            self.wdg_hits += 1
            return self.wdg_cache[key]
        else:
            self.wdg_misses += 1
            return None

    def store_wdg(self, constraints, hval):
        """Store final WDG heuristic value for the full constraint set."""
        key = self._hash_constraints_global(constraints)
        self.wdg_cache[key] = int(hval)
    
    def print_stats(self):
        """Print cache statistics."""
        print(f"\n{'='*60}")
        print("Cache Statistics")
        print(f"{'='*60}")
        
        mdd_total = self.mdd_hits + self.mdd_misses
        if mdd_total > 0:
            mdd_rate = self.mdd_hits / mdd_total * 100
            print(f"MDD Cache:")
            print(f"  Hits: {self.mdd_hits}, Misses: {self.mdd_misses}")
            print(f"  Hit rate: {mdd_rate:.1f}%")
        
        dep_total = self.dep_hits + self.dep_misses
        if dep_total > 0:
            dep_rate = self.dep_hits / dep_total * 100
            print(f"\nDependency Cache:")
            print(f"  Hits: {self.dep_hits}, Misses: {self.dep_misses}")
            print(f"  Hit rate: {dep_rate:.1f}%")
        
        delta_total = self.delta_hits + self.delta_misses
        if delta_total > 0:
            delta_rate = self.delta_hits / delta_total * 100
            print(f"\nDelta Cache:")
            print(f"  Hits: {self.delta_hits}, Misses: {self.delta_misses}")
            print(f"  Hit rate: {delta_rate:.1f}%")

        wdg_total = self.wdg_hits + self.wdg_misses
        if wdg_total > 0:
            wdg_rate = self.wdg_hits / wdg_total * 100
            print(f"\nWDG Heuristic Cache:")
            print(f"  Hits: {self.wdg_hits}, Misses: {self.wdg_misses}")
            print(f"  Hit rate: {wdg_rate:.1f}%")
        
        print(f"\nTotal cache entries:")
        print(f"  MDDs: {len(self.mdd_cache)}")
        print(f"  Dependencies: {len(self.dependency_cache)}")
        print(f"  Deltas: {len(self.delta_cache)}")
        print(f"  WDG heuristics: {len(self.wdg_cache)}")
        print(f"{'='*60}\n")
    
    def clear(self):
        """Clear all caches."""
        self.mdd_cache.clear()
        self.dependency_cache.clear()
        self.delta_cache.clear()
        self.wdg_cache.clear()
        
        self.mdd_hits = 0
        self.mdd_misses = 0
        self.dep_hits = 0
        self.dep_misses = 0
        self.delta_hits = 0
        self.delta_misses = 0
        self.wdg_hits = 0
        self.wdg_misses = 0
