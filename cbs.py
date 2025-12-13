import time as timer
import random
import heapq
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost, get_location
import copy

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
        # heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        heapq.heappush(self.open_list, (node['cost'] + node['h'], len(node['collisions']), self.num_of_generated, node))
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        # print("Expanded nodes {}".format(self.num_of_expanded))
        return node

    def find_solution(self, disjoint=True):
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

        root['h'] = 0
        root['collisions'] = self.detect_collisions(root['paths'])
        self.push_node(root)

        while len(self.open_list) > 0:
            curr = self.pop_node()

            if not curr['collisions']:
                self.print_results(curr)
                self.write_results()
                return curr['paths'] # this is the goal node
            
            collision = curr['collisions'][0]
            constraints = self.disjoint_splitting(collision)
            for constraint in constraints:
                if self.is_conflicting_constraint(constraint, curr['constraints']):
                    continue
                child = {}
                child['constraints'] = copy.deepcopy(curr['constraints'])
                if constraint not in child['constraints']:
                    child['constraints'].append(constraint)
                child['paths']= copy.deepcopy(curr['paths'])

                prune_child = False
                if constraint['positive']:
                    conflicted_agents = self.paths_violate_constraint(constraint, child['paths'])
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
                    child['collisions'] = self.detect_collisions(child['paths'])
                    child['cost'] = get_sum_of_cost(child['paths'])
                    child['h'] = 0
                    self.push_node(child)

        self.print_results(root)
        self.write_results()        
        return root['paths']

    def detect_collision(self, path1, path2):
        ##############################
        # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
        #           There are two types of collisions: vertex collision and edge collision.
        #           A vertex collision occurs if both robots occupy the same location at the same timestep
        #           An edge collision occurs if the robots swap their location at the same timestep.
        #           You should use "get_location(path, t)" to get the location of a robot at time t.
        for timestep in range(1, max(len(path1), len(path2))):
            if get_location(path1, timestep) == get_location(path2, timestep):
                # vertex collision
                return {
                    'loc': [get_location(path1, timestep)],
                    'timestep': timestep
                    }
            if get_location(path1, timestep) == get_location(path2, timestep-1) and get_location(path2, timestep) == get_location(path1, timestep-1):
                # edge collision
                return {
                    'loc': [get_location(path1, timestep-1), get_location(path1, timestep)],
                    'timestep': timestep
                    }
            
        return None

    def detect_collisions(self, paths):
        ##############################
        # Return a list of first collisions between all robot pairs.
        # A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
        # causing the collision, and the timestep at which the collision occurred.

        collisions = []
        num_paths = len(paths)
        for agent1 in range(num_paths-1):
            for agent2 in range(agent1+1, num_paths):
                collision = self.detect_collision(paths[agent1], paths[agent2])
                if collision != None:
                    collision['a1'] = agent1
                    collision['a2'] = agent2
                    collisions.append(collision)
        return collisions

    def standard_splitting(self, collision):
        # Return a list of (two) constraints to resolve the given collision
        # Vertex collision: the first constraint prevents the first agent to be at the specified location at the
        #                   specified timestep, and the second constraint prevents the second agent to be at the
        #                   specified location at the specified timestep.
        # Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
        #                 specified timestep, and the second constraint prevents the second agent to traverse the
        #                 specified edge at the specified timestep

        if len(collision['loc']) == 1:
            # vertex collision:
            constraint1 = {
                'agent': collision['a1'],
                'loc': collision['loc'],
                'timestep': collision['timestep'],
                'positive': False
            }
            constraint2 = {
                'agent': collision['a2'],
                'loc': collision['loc'],
                'timestep': collision['timestep'],
                'positive': False
            }
        else:
            # edge collision
            constraint1 = {
                'agent': collision['a1'],
                'loc': [collision['loc'][0], collision['loc'][1]],
                'timestep': collision['timestep'],
                'positive': False
            }
            constraint2 = {
                'agent': collision['a2'],
                'loc': [collision['loc'][1], collision['loc'][0]],
                'timestep': collision['timestep'],
                'positive': False
            }
        return [constraint1, constraint2]

    def disjoint_splitting(self, collision):
        # Return a list of (two) constraints to resolve the given collision
        # Vertex collision: the first constraint enforces one agent to be at the specified location at the
        #                   specified timestep, and the second constraint prevents the same agent to be at the
        #                   same location at the timestep.
        # Edge collision: the first constraint enforces one agent to traverse the specified edge at the
        #                 specified timestep, and the second constraint prevents the same agent to traverse the
        #                 specified edge at the specified timestep

        agent = 'a1'
        if random.randint(0, 1):
            agent = 'a2'
        
        if len(collision['loc']) == 1:
            constraint1 = {
                'agent': collision[agent],
                'loc': collision['loc'],
                'timestep': collision['timestep'],
                'positive': True
            }
            constraint2 = {
                'agent': collision[agent],
                'loc': collision['loc'],
                'timestep': collision['timestep'],
                'positive': False
            }
        else:
            if agent == 'a1':
                constraint1 = {
                    'agent': collision[agent],
                    'loc': [collision['loc'][0], collision['loc'][1]],
                    'timestep': collision['timestep'],
                    'positive': True
                }
                constraint2 = {
                    'agent': collision[agent],
                    'loc': [collision['loc'][0], collision['loc'][1]],
                    'timestep': collision['timestep'],
                    'positive': False
                }
            else:
                constraint1 = {
                    'agent': collision[agent],
                    'loc': [collision['loc'][1], collision['loc'][0]],
                    'timestep': collision['timestep'],
                    'positive': True
                }
                constraint2 = {
                    'agent': collision[agent],
                    'loc': [collision['loc'][1], collision['loc'][0]],
                    'timestep': collision['timestep'],
                    'positive': False
                }            
        return [constraint1, constraint2]    
        
    def paths_violate_constraint(self, constraint, paths):
        assert constraint['positive'] is True
        rst = []
        for i in range(len(paths)):
            if i == constraint['agent']:
                continue
            curr = get_location(paths[i], constraint['timestep'])
            prev = get_location(paths[i], constraint['timestep'] - 1)
            if len(constraint['loc']) == 1:  # vertex constraint
                if constraint['loc'][0] == curr:
                    rst.append(i)
            else:  # edge constraint
                if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                        or constraint['loc'] == [curr, prev]:
                    rst.append(i)
        return rst

    def is_conflicting_constraint(self, new_constraint, constraints):
        # Returns true if the constraint we want to add conflicts with an existing constraint.
        if new_constraint in constraints:
            return True

        t = new_constraint['timestep']
        constraints_at_t = [c for c in constraints if c['timestep'] == t and c['agent'] == new_constraint['agent']]
        is_new_vertex_constraint = False
        if len(new_constraint['loc']) == 1:
            is_new_vertex_constraint = True

        for old_constraint in constraints_at_t:
            if len(old_constraint['loc']) == 1:
                # old vertex constraint
                if old_constraint['positive']:
                    # old positive vertex constraint (old says you must be somewhere)
                    if is_new_vertex_constraint and not new_constraint['positive'] and new_constraint['loc'] == old_constraint['loc']:
                        # new negative vertex constraint (new says you cant be there)
                        return True
                    if is_new_vertex_constraint and new_constraint['positive'] and new_constraint['loc'] != old_constraint['loc']:
                        # new positive vertex constraint (new says you must be somewhere else)
                        return True
                    if not is_new_vertex_constraint and new_constraint['positive'] and new_constraint['loc'][1] != old_constraint['loc'][0]:
                        # new positive edge constraint (new says you must move somewhere else)
                        return True
                else:
                    # old negative vertex constraint (old says you cant be at a spot)
                    if is_new_vertex_constraint and new_constraint['positive'] and new_constraint['loc'] == old_constraint['loc']:
                        # new positive vertex constraint (new says you must be there)
                        return True
                    if not is_new_vertex_constraint and new_constraint['positive'] and new_constraint['loc'][1] == old_constraint['loc'][0]:
                        # new positive edge constraint (new says you must move there)
                        return True          
            else:
                # old edge constraint
                if old_constraint['positive']:
                    # old positive edge constraint (old says you must move somewhere)
                    if is_new_vertex_constraint and new_constraint['positive'] and new_constraint['loc'][0] != old_constraint['loc'][1]:
                        # new positive vertex constraint (new says you must be somewhere else)
                        return True
                    if is_new_vertex_constraint and not new_constraint['positive'] and new_constraint['loc'][0] == old_constraint['loc'][1]:
                        # new negative vertex constraint (new says you cant be at that destination)
                        return True
                    if not is_new_vertex_constraint and new_constraint['positive']:
                        # new positive edge constraint (new says you have a different edge to move)
                        return True
                    if not is_new_vertex_constraint and not new_constraint['positive'] and new_constraint['loc'] == old_constraint['loc']:
                        # new negative edge constraint (new says you cant make that move)
                        return True
                else:
                    # old negative edge constraint (old says you cant make this move)
                    if not is_new_vertex_constraint and new_constraint['positive'] and new_constraint['loc'] == old_constraint['loc']:
                        # new positive edge constraint
                        return True
        return False

    def write_results(self):
        filename = 'results.csv'
        file = open(filename, 'a')
        generated = self.num_of_generated
        expanded = self.num_of_expanded
        time = CPU_time = timer.time() - self.start_time
        num_open_cells = 0
        for row in self.my_map:
            num_open_cells += len(row)-sum(row)
        agents = self.num_of_agents
        density = agents / num_open_cells
        num_cols = len(self.my_map[0])
        num_rows = len(self.my_map)
        res = f'{num_cols}, {num_rows}, {agents}, {generated}, {expanded}, {density:.3f}, {round(time,3)}\n'
        file.write(res)
        file.close()

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        # print("Final constraints:", node['constraints'])
        for i in range(len(node['paths'])):
            print("Agent {}: {}".format(i, node['paths'][i]))
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
