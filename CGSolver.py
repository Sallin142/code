import time as timer
from mdd import buildMDDTree, get_all_optimal_paths, check_MDDs_for_conflict, balanceMDDs
from mvc import Graph
from single_agent_planner import a_star, get_sum_of_cost
from cbs import CBSSolver, detect_collisions, disjoint_splitting, paths_violate_constraint
import copy


class CGSolver(CBSSolver):
    def get_cg_heuristic(self, grid, paths, starts, goals, hvals, constraints, cache=[], mdds=[]):
        cardinals = []

        if (len(cache) == 0 and len(mdds) == 0):
            for i in range(len(paths)):
                optimal = get_all_optimal_paths(grid, starts[i], goals[i], hvals[i], i, constraints)
                if (optimal == []):
                    return -1
                _, nodes = buildMDDTree(optimal)
                cache.append(optimal)
                mdds.append(nodes)

        for i in range(len(paths)):
            first = cache[i]
            dict1 = mdds[i]

            for j in range(i+1,len(paths)):

                second = cache[j] 
                dict2 = mdds[j] 
                balanceMDDs(first, second, dict1, dict2)
                
                if (check_MDDs_for_conflict(dict1, dict2)):
                    cardinals.append((i,j))
        
        g = Graph(len(paths))
        for conflict in cardinals:
            g.addEdge(conflict[0], conflict[1])
        cover = g.getVertexCover()
        return len(cover) 

    def find_solution(self, disjoint=True, initial=[], heuristic=0, logging = True):

        self.timer = timer.time()

        root = {'cost': 0,
                'h': heuristic,
                'constraints': initial,
                'paths': [],
                'collisions': []}
        for i in range(self.agents):
            path = a_star(self.grid, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['h'] = self.get_cg_heuristic(self.grid, root['paths'], self.starts, self.goals, self.heuristics, root['constraints'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        while len(self.queue) > 0:
            curr = self.pop_node()

            if not curr['collisions']:
                if(logging):
                    self.print_results(curr)
                return curr['paths']
            
            collision = curr['collisions'][0]
            constraints = disjoint_splitting(collision)
            for constraint in constraints:
                blocked = False
                if constraint in curr['constraints']:
                    blocked = True
                else:
                    t = constraint['timestep']
                    matching = [c for c in curr['constraints'] if c['timestep'] == t and c['agent'] == constraint['agent']]
                    vertex = False
                    if len(constraint['loc']) == 1:
                        vertex = True

                    for prev in matching:
                        if len(prev['loc']) == 1:
                            if prev['positive']:
                                if vertex and not constraint['positive'] and constraint['loc'] == prev['loc']:
                                    blocked = True
                                    break
                                if vertex and constraint['positive'] and constraint['loc'] != prev['loc']:
                                    blocked = True
                                    break
                                if not vertex and constraint['positive'] and constraint['loc'][1] != prev['loc'][0]:
                                    blocked = True
                                    break
                            else:
                                if vertex and constraint['positive'] and constraint['loc'] == prev['loc']:
                                    blocked = True
                                    break
                                if not vertex and constraint['positive'] and constraint['loc'][1] == prev['loc'][0]:
                                    blocked = True
                                    break
                        else:
                            if prev['positive']:
                                if vertex and constraint['positive'] and constraint['loc'][0] != prev['loc'][1]:
                                    blocked = True
                                    break
                                if vertex and not constraint['positive'] and constraint['loc'][0] == prev['loc'][1]:
                                    blocked = True
                                    break
                                if not vertex and constraint['positive']:
                                    blocked = True
                                    break
                                if not vertex and not constraint['positive'] and constraint['loc'] == prev['loc']:
                                    blocked = True
                                    break
                            else:
                                if not vertex and constraint['positive'] and constraint['loc'] == prev['loc']:
                                    blocked = True
                                    break
                
                if blocked:
                    continue
                child = {}
                child['constraints'] = copy.deepcopy(curr['constraints'])
                if constraint not in child['constraints']:
                    child['constraints'].append(constraint)
                child['paths']= copy.deepcopy(curr['paths'])

                skip = False
                if constraint['positive']:
                    affected = paths_violate_constraint(constraint, child['paths'])
                    for i in affected:
                        route = a_star(self.grid, self.starts[i], self.goals[i], self.heuristics[i],
                            i, child['constraints'])
                        if route is None:
                            skip = True
                            break
                        else:
                            child['paths'][i] = route
                if skip:
                    continue

                agent = constraint['agent']
                path = a_star(self.grid, self.starts[agent], self.goals[agent], self.heuristics[agent],
                          agent, child['constraints'])
                if path is not None:
                    child['paths'][agent] = path
                    child['collisions'] = detect_collisions(child['paths'])
                    child['cost'] = get_sum_of_cost(child['paths'])
                    child['h'] = self.get_cg_heuristic(self.grid, child['paths'], self.starts, self.goals, self.heuristics, child['constraints'])

                    self.push_node(child)

        if(logging):
            self.print_results(root)
        return root['paths']
