import time as timer
from mdd import buildMDDTree, get_all_optimal_paths, check_MDDs_for_conflict, balanceMDDs
from mvc import Graph
from single_agent_planner import a_star, get_sum_of_cost
from cbs import CBSSolver
import copy

class CGSolver(CBSSolver):
    def get_cg_heuristic(self, my_map, paths, starts, goals, low_level_h, constraints, all_paths=None, all_mdds=None):
        if all_paths is None:
            all_paths = []
        if all_mdds is None:
            all_mdds = []

        cardinal_conflicts = []

        # build mdd for every agent
        # store reuse it later
        # all_paths = []
        # all_mdds = []

        if (len(all_paths) == 0 and len(all_mdds) == 0):
            for i in range(len(paths)):
                newpaths = get_all_optimal_paths(my_map, starts[i], goals[i], low_level_h[i], i, constraints)
                if (newpaths == []):
                    return -1
                _, nodes_dict = buildMDDTree(newpaths)
                all_paths.append(newpaths)
                all_mdds.append(nodes_dict)

        for i in range(len(paths)): # num of agents in map
            paths1 = all_paths[i]
            nodes_dict1 = all_mdds[i]

            for j in range(i+1,len(paths)):

                paths2 = all_paths[j] 
                nodes_dict2 = all_mdds[j] 
                balanceMDDs(paths1, paths2, nodes_dict1, nodes_dict2)
                
                if (check_MDDs_for_conflict(nodes_dict1, nodes_dict2)):
                    cardinal_conflicts.append((i,j))
        # print("Cardinal conflicts found:", cardinal_conflicts)
        
        g = Graph(len(paths))
        for conflict in cardinal_conflicts:
            g.addEdge(conflict[0], conflict[1])
        vertex_cover = g.getVertexCover()
        return len(vertex_cover) 

    def find_solution(self, disjoint=True, root_constraints=[], root_h=0, record_results = True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        root = {'cost': 0,
                'h': root_h,
                'constraints': root_constraints,
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['h'] = self.get_cg_heuristic(self.my_map, root['paths'], self.starts, self.goals, self.heuristics, root['constraints'])
        root['collisions'] = super().detect_collisions(root['paths'])
        self.push_node(root)

        while len(self.open_list) > 0:
            curr = self.pop_node()

            if not curr['collisions']:
                if(record_results):
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
                    child['h'] = self.get_cg_heuristic(self.my_map, child['paths'], self.starts, self.goals, self.heuristics, child['constraints'])

                    self.push_node(child)

        if(record_results):
            self.print_results(root)
            self.write_results()
        return root['paths']
