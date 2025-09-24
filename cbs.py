import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost


def detect_collision(path1, path2):
    time_range = max(len(path1), len(path2))
    for time in range(time_range):
        loc_c1 = get_location(path1, time)
        loc_c2 = get_location(path2, time)
        loc1 = get_location(path1, time + 1)
        loc2 = get_location(path2, time + 1)
        if loc1 == loc2:
            return [loc1], time
        if [loc_c1, loc1] == [loc2, loc_c2]:
            return [loc2, loc_c2], time
        
       
    return None


def detect_collisions(paths):
    collisions = []
    for i in range(len(paths) - 1):
        for j in range(i + 1, len(paths)):
            if detect_collision(paths[i], paths[j]) != None:
                position, t = detect_collision(paths[i], paths[j])
                collisions.append({'a1': i,'a2': j,'loc': position,'timestep': t + 1})
    return collisions


def standard_splitting(collision):
    constraints = []
    if len(collision['loc']) == 1:
        constraints.append({'agent': collision['a1'],'loc': collision['loc'],'timestep': collision['timestep'],'positive': False})
        constraints.append({'agent': collision['a2'],'loc': collision['loc'],'timestep': collision['timestep'],'positive': False})
    else:
        constraints.append({'agent': collision['a1'],'loc': [collision['loc'][0], collision['loc'][1]],'timestep': collision['timestep'],'positive': False})
        constraints.append({'agent': collision['a2'],'loc': [collision['loc'][1], collision['loc'][0]],'timestep': collision['timestep'],'positive': False})
    return constraints

def disjoint_splitting(collision):
    constraints = []
    agent = random.randint(0, 1)
    a = 'a' + str(agent + 1)
    if len(collision['loc']) == 1:
        constraints.append({'agent': collision[a], 'loc': collision['loc'],'timestep': collision['timestep'],'positive': True})
        constraints.append({'agent': collision[a], 'loc': collision['loc'],'timestep': collision['timestep'],'positive': False})
    else:
        if agent == 0:
            constraints.append({'agent': collision[a],'loc': [collision['loc'][0], collision['loc'][1]],'timestep': collision['timestep'],'positive': True})
            constraints.append({'agent': collision[a],'loc': [collision['loc'][0], collision['loc'][1]],'timestep': collision['timestep'],'positive': False})
        else:
            constraints.append({'agent': collision[a],'loc': [collision['loc'][1], collision['loc'][0]],'timestep': collision['timestep'],'positive': True})
            constraints.append({'agent': collision[a],'loc': [collision['loc'][1], collision['loc'][0]],'timestep': collision['timestep'],'positive': False})
    return constraints

def paths_violate_constraint(constraint, paths):
    assert constraint['positive'] is True
    violating_agents = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:
            if constraint['loc'][0] == curr:
                violating_agents.append(i)
        else: 
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                violating_agents.append(i)
    return violating_agents


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
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
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)
        
        while len(self.open_list) > 0:
            parent_node = self.pop_node()
            if parent_node['collisions'] == []:
                self.print_results(parent_node)
                for agent_path in parent_node['paths']:
                    print(agent_path)
                return parent_node['paths']
            collision = parent_node['collisions'].pop(0)
            constraints = disjoint_splitting(collision) if disjoint else standard_splitting(collision)

            for constraint in constraints:
                child_node = {'cost': 0,
                    'constraints': [constraint],
                    'paths': [],
                    'collisions': []
                }
                for existing_constraint in parent_node['constraints']:
                    if existing_constraint not in child_node['constraints']:
                        child_node['constraints'].append(existing_constraint)
                for agent_path in parent_node['paths']:
                    child_node['paths'].append(agent_path)
                
                ai = constraint['agent']
                path = a_star(self.my_map, self.starts[ai], self.goals[ai], self.heuristics[ai], ai, child_node['constraints'])

                if path is not None:
                    child_node['paths'][ai] = path
                    continue_flag = False
                    if constraint['positive']:
                        vol = paths_violate_constraint(constraint, child_node['paths'])
                        for v in vol:
                            path_v = a_star(self.my_map, self.starts[v], self.goals[v], self.heuristics[v], v, child_node['constraints'])
                            if path_v is None:
                                continue_flag = True
                            else:
                                child_node['paths'][v] = path_v
                        if continue_flag:
                            continue
                    child_node['collisions'] = detect_collisions(child_node['paths'])
                    child_node['cost'] = get_sum_of_cost(child_node['paths'])
                    self.push_node(child_node)
        return None

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
