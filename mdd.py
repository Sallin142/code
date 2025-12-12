import heapq
import queue
from single_agent_planner import build_constraint_table, is_constrained
"Multi-Value Decision Diagram (MDD) for Multi-Agent Path Finding (MAPF)"


class NodeDictWrap(dict):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        return super().__getitem__(item)

    def __setattr__(self, item, value):
        return super().__setitem__(item, value)

    def __lt__(self,other):
        return self.h_val < other.h_val


def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def compute_heuristics(my_map, goal):
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root

    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values

def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path

def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], NodeDictWrap(node)))

def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr

def compare_nodes(n1, n2):
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

def get_all_optimal_paths(my_map, start_loc, goal_loc, h_values, agent, constraints, max_paths = 500):
    """
    find all optimal paths from start to goal to prevent exponential explosion in empty map, set max_path to 500
    
    :param my_map: 2d grid map
    :param start_loc: start location(x,y)
    :param goal_loc: goal location(x,y)
    :param h_values: heuristic value
    :param agent: agent id
    :param constraints: list of constraints to avoid
    :param max_paths: to prevent exponential explosion in empty map, set max_path to 500
    """

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    first_time = True
    paths = []
    h_value = h_values[start_loc]
    constraint_table = build_constraint_table(constraints, agent)
    
    # build positive constraint table
    pos_constraint_table = dict()
    for constraint in constraints:
        if constraint['agent'] == agent:
            continue
        if constraint['positive'] == False:
            continue

        # for positive constraint 
        timestep = constraint['timestep']
        # vertext constraint
        if timestep in pos_constraint_table:
            pos_constraint_table[timestep].append(constraint)
        # edge constraint
        else:
            pos_constraint_table[timestep] = [constraint]
    
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'time': 0}
    push_node(open_list, root)
    closed_list[(root['loc'], root['time'])] = root
    
    iterations = 0
    optimal_f_value = None
    
    while len(open_list) > 0:
        iterations += 1
        curr = pop_node(open_list)
        
        # stop when we find the optimal cost and current node has higher f
        if optimal_f_value is not None:
            curr_f = curr['g_val'] + curr['h_val']
            if curr_f > optimal_f_value:
                return paths
        
        # goal test
        if curr['loc'] == goal_loc:
            curr_path = get_path(curr)
            if(first_time):
                earliest_goal_timestep = len(curr_path)
                optimal_f_value = curr['g_val'] 
                first_time = False
            elif (len(curr_path) > earliest_goal_timestep):
                # this path is longer than optimal
                return paths
            paths.append(curr_path)
            
            # limit number of paths to prevent exponential explosion
            if len(paths) >= max_paths:
                return paths
            
            continue


        next_loc = None
        next_time = curr['time'] + 1
        if next_time in constraint_table:
            constraints = constraint_table[next_time]
            for constraint in constraints:
                if not constraint['positive']:
                    continue

                location = constraint['loc']
                if len(location) == 1:
                    next_loc = location[0]
                    break
                else:
                    if curr['loc'] == location[0]:
                        next_loc = location[1]
                        break
                    else:
                        next_loc = ()
                        break
        
        if next_loc == ():
            continue

        if next_loc != None:

            pos_constrained = False
            check_time = curr['time'] + 1
            if check_time in pos_constraint_table:
                constraints = pos_constraint_table[check_time]
                for constraint in constraints:
                    location = constraint['loc']
                    if len(location) == 1:
                        if location[0] == next_loc:
                            pos_constrained = True
                            break
                    else:
                        if location[0] == next_loc and location[1] == curr['loc']:
                            pos_constrained = True
                            break
            
            if pos_constrained:
                continue

            if next_loc[0] < 0 or next_loc[0] >= len(my_map) \
                or next_loc[1] < 0 or next_loc[1] >= len(my_map[0]):
                return paths
            if my_map[next_loc[0]][next_loc[1]]:
                return paths

            possible_locs = []
            for dir in range(5): 
                possible_locs.append(move(curr['loc'], dir))
            if next_loc not in possible_locs:
                continue

            child = {'loc': next_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[next_loc],
                    'parent': curr,
                    'time': curr['time'] + 1}

            if (child['loc'], child['time']) in closed_list:
                existing_node = closed_list[(child['loc'], child['time'])]
                if child['g_val'] + child['h_val'] <= existing_node['g_val'] + existing_node['h_val']:
                    closed_list[(child['loc'], child['time'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['time'])] = child
                push_node(open_list, child)
            continue

        expanded_count = 0
        for dir in range(5):
            child_loc = move(curr['loc'], dir)
            
            # check bounds
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
                continue
            
            # check walls
            if my_map[child_loc[0]][child_loc[1]]:
                continue

            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr,
                    'time': curr['time'] + 1}

            pos_constrained = False
            check_time = curr['time'] + 1
            if check_time in pos_constraint_table:
                constraints = pos_constraint_table[check_time]
                for constraint in constraints:
                    location = constraint['loc']
                    if len(location) == 1:
                        if location[0] == child['loc']:
                            pos_constrained = True
                            break
                    else:
                        if location[0] == child['loc'] and location[1] == curr['loc']:
                            pos_constrained = True
                            break
            
            if pos_constrained:
                continue

            constrained_result = is_constrained(curr['loc'], child['loc'], child['time'], constraint_table)
            if constrained_result == 0:
                continue
            
            expanded_count += 1
            
            if (child['loc'], child['time']) in closed_list:
                existing_node = closed_list[(child['loc'], child['time'])]
                if child['g_val'] + child['h_val'] <= existing_node['g_val'] + existing_node['h_val']:
                    closed_list[(child['loc'], child['time'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['time'])] = child
                push_node(open_list, child)
        
    return paths

class MDDNode():

    def __init__(self, location, timestep):
        self.parent = []
        self.children = []
        self.location = location
        self.timestep = timestep

    def display(self):
        print("Parent:", [o.location for o in self.parent])
        print("Location:", self.location)
        print("Children:", [o.location for o in self.children])
        print("Timestep:", self.timestep)
        print(self)

    def updateNode(self, parent):
        if (parent not in self.parent):
            self.parent.append(parent)
        if (self not in parent.children):
            parent.children.append(self)
    
class JointMDDNode():

    def __init__(self, location1, location2, timestep):
        self.parent = []
        self.children = []
        self.location = [location1, location2]
        self.timestep = timestep

    def display(self):
        print("Timestep:", self.timestep)
        print("Parent:", [o.location for o in self.parent])
        print("Location:", self.location)
        print("Children:", [o.location for o in self.children])
        print(self)

    def updateNode(self, parent):
        if (parent not in self.parent):
            self.parent.append(parent)
        if (self not in parent.children):
            parent.children.append(self)

def displayLayer(node, level):
    if(len(node.children) == 0):
        return
    else:
        for child in node.children:
            displayLayer(child, level+1)
   
def buildMDDTree(optimal_paths ):
    """
    Build mdd tree from optimal paths (ditected acyclic graph)

    Args:
        paths: List of paths, where each path is a list of locations
    
    Returns:
        (root_node, nodes_dict) where:
        - root_node: The root MDDNode
        - nodes_dict: Dictionary mapping (location, timestep) -> MDDNode
    """
    if not optimal_paths:
        return None, {}
    
    root_location = optimal_paths[0][0]
    root_node = MDDNode(root_location , 0)

    existing_nodes = {
        (root_location, 0): root_node
    }

    for path in optimal_paths:
        curr = root_node
        for i in range(1, len(path)):
            location = path[i]
            
            # check if node already exists at this (location, timestep)
            if (location, i) in existing_nodes:
                new_node = existing_nodes[(location, i)]
            else:
                new_node = MDDNode(location, i)
                existing_nodes[(location, i)] = new_node
            
            # add edge from curr to new_node
            new_node.updateNode(curr)
            curr = new_node

    return root_node, existing_nodes

def extendMDDTree(goal_node, height_diff, node_dict):
    """
    Extend MDD by adding wait actions at goal.
    """
    curr = goal_node
    for i in range(height_diff):
        new_node = MDDNode(goal_node.location, curr.timestep + 1)
        new_node.updateNode(curr)
        node_dict[(goal_node.location, curr.timestep + 1)] = new_node
        curr = new_node

def check_MDDs_for_conflict(node_dict1, node_dict2):
    """
    Check if two MDDs have a cardinal conflict.
    """
    dict1 = {}
    dict2 = {}
    
    # Group nodes by timestep
    for (loc, time) in node_dict1.keys():
        if time in dict1:
            dict1[time].append(loc)
        else: 
            dict1[time] = [loc]

    for (loc, time) in node_dict2.keys():
        if time in dict2:
            dict2[time].append(loc)
        else:
            dict2[time] = [loc]

    # balance mdds
    diff = abs(len(dict1) - len(dict2))
    if len(dict1) == len(dict2):
        pass
    elif len(dict1) > len(dict2):
        last_timestep = max(dict2.keys())
        for i in range(diff):
            dict2[last_timestep + i + 1] = dict2[last_timestep]
    else:
        last_timestep = max(dict1.keys())
        for i in range(diff):
            dict1[last_timestep + i + 1] = dict1[last_timestep]

    # check for cardinal conflict
    for time in dict1.keys():
        if time not in dict2:
            continue
        locs1 = dict1[time]
        locs2 = dict2[time]
        
        # both must have exactly one location
        if len(locs1) == 1 and len(locs2) == 1:
            # and they must be the same location
            if locs1[0] == locs2[0]:
                return True

    return False

def balanceMDDs(paths1, paths2, node_dict1, node_dict2):
    """
    balance two MDDs to have the same height by adding wait actions.
    """
    height1 = len(paths1[0])
    height2 = len(paths2[0])

    if (height1 != height2):
        if (height1 < height2):
            goal_loc = paths1[0][-1]
            bottom_node = node_dict1[(goal_loc, height1-1)]
            extendMDDTree(bottom_node, height2-height1, node_dict1)
            
        else:
            goal_loc = paths2[0][-1]
            bottom_node = node_dict2[(goal_loc, height2-1)]
            extendMDDTree(bottom_node, height1-height2, node_dict2)

def buildJointMDD(paths1, paths2, root1, node_dict1, root2, node_dict2 ):
    """
    Build joint MDD from two individual MDDs
    """
    height1 = len(paths1[0])
    height2 = len(paths2[0])

    # balance MDDs to same height
    if (height1 != height2):
        if (height1 < height2):
            goal_loc = paths1[0][-1]
            bottom_node = node_dict1[(goal_loc, height1-1)]
            extendMDDTree(bottom_node, height2-height1 , node_dict1)
        else:
            goal_loc = paths2[0][-1]
            bottom_node = node_dict2[(goal_loc, height2-1)]
            extendMDDTree(bottom_node, height1-height2, node_dict2)

    q = queue.Queue()
    root = JointMDDNode(root1.location, root2.location, 0)
    q.put(root)

    existing_dict = {
        (root1.location, root2.location, 0): root
    }
    
    # track the deepest node we can reach
    deepest_node = root
    
    while (not q.empty()):
        curr = q.get()
        
        # update deepest node
        if curr.timestep > deepest_node.timestep:
            deepest_node = curr

        loc1, time1 = curr.location[0], curr.timestep
        loc2, time2 = curr.location[1], curr.timestep

        node1 = node_dict1[(loc1, time1)]
        node2 = node_dict2[(loc2, time2)]

        # if either agent reached the end, stop exploring this branch
        if (len(node1.children) == 0 or len(node2.children) == 0):
            continue
        
        # get all possible combinations of next locations
        loc_combinations = []
        for node1_child in node1.children:
            for node2_child in node2.children:
                loc_combinations.append((node1_child.location, node2_child.location))

        # build joint children
        for combo in loc_combinations:
            next_loc1, next_loc2 = combo[0], combo[1]
            
            # check for vertex conflict
            if next_loc1 == next_loc2:
                continue
            
            # check for edge conflict
            if loc1 == next_loc2 and loc2 == next_loc1:
                continue
            
            params = (next_loc1, next_loc2, curr.timestep + 1)
            if params in existing_dict:
                new_node = existing_dict[params ]
            else:
                new_node = JointMDDNode( next_loc1, next_loc2, curr.timestep + 1)
                existing_dict[params] = new_node
                q.put(new_node)
            
            new_node.updateNode(curr)
    
    return root, deepest_node

def check_jointMDD_for_dependency(bottom_node, path1, path2):
    """
    Check if two agents are dependent based on their joint MDD
    
    :param bottom_node: deepest node in joint MDD
    :param path1: Optimal path for agent 1
    :param path2: Optimal path for agent 2
    """

    optimal_time = max(len(path1), len(path2)) - 1
    goal_loc1 = path1[-1]
    goal_loc2 = path2[-1]
    
    # check if bottom node reaches goals at optimal time
    if (bottom_node.location[0] != goal_loc1 or 
        bottom_node.location[1] != goal_loc2 or 
        bottom_node.timestep != optimal_time):
        return True
    
    return False  