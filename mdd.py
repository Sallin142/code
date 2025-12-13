import heapq
import queue
from single_agent_planner import check_agent_pos_constrained, is_pos_constrained, is_constrained

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
            if child_loc[0] < 0 or child_loc[0] >= len(my_map)               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
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
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

def get_all_optimal_paths(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
    """
    open_list = []
    closed_list = dict() 
    earliest_goal_timestep = 0
    first_time = True
    paths = []
    h_value = h_values[start_loc]
    # Build constraint table
    constraint_table = dict()
    for constraint in constraints:
        if constraint['agent'] != agent:
            continue
        t = constraint['timestep']
        if t in constraint_table:
            constraint_table[t].append(constraint)
        else:
            constraint_table[t] = [constraint]
    
    # Build positive constraint table
    pos_constraint_table = dict()
    for constraint in constraints:
        if constraint['agent'] == agent:
            continue
        if constraint['positive'] == False:
            continue
        t = constraint['timestep']
        if t in pos_constraint_table:
            pos_constraint_table[t].append(constraint)
        else:
            pos_constraint_table[t] = [constraint]    
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'time': 0}
    push_node(open_list, root)
    closed_list[(root['loc'], root['time'])] = root
    while len(open_list) > 0:
        
        curr = pop_node(open_list)
        
        if curr['loc'] == goal_loc:
            curr_path = get_path(curr)
            
            if(first_time):
                earliest_goal_timestep = len(curr_path)
                first_time = False
            elif (len(curr_path) > earliest_goal_timestep):
                return paths
            paths.append(curr_path)
            continue

        next_loc = check_agent_pos_constrained(curr, constraint_table)
        if next_loc == ():
            continue

        if next_loc != None:
            if is_pos_constrained(curr['loc'], next_loc, curr['time'] + 1, pos_constraint_table):
                
                continue

            if next_loc[0] < 0 or next_loc[0] >= len(my_map)                or next_loc[1] < 0 or next_loc[1] >= len(my_map[0]):
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
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['time'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['time'])] = child
                push_node(open_list, child)
            continue

        for dir in range(5):
            child_loc = move(curr['loc'], dir)
            if child_loc[0] < 0 or child_loc[0] >= len(my_map)               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue            
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr,
                    'time': curr['time'] + 1}

            if is_pos_constrained(curr['loc'], child['loc'], child['time'], pos_constraint_table):
                
                continue

            if is_constrained(curr['loc'], child['loc'], child['time'], constraint_table):
                
                continue
            
            if (child['loc'], child['time']) in closed_list:
                existing_node = closed_list[(child['loc'], child['time'])]
                if compare_nodes(child, existing_node):
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
   
def buildMDDTree(optimal_paths):
    root_location = optimal_paths[0][0]
    root_node = MDDNode(root_location, 0)
    

    existing_nodes = {
        (root_location, 0): root_node
    }

    for path in optimal_paths:
        curr = root_node
        
        for i in range(1, len(path)):
            location = path[i]
            new_node = MDDNode(location, i)
            existing_nodes[(location, i)] = new_node
            new_node.updateNode(curr)
            curr = new_node

    return root_node, existing_nodes

def extendMDDTree(goal_node, height_diff, node_dict):
    curr = goal_node
    for i in range(height_diff):
        
                
        new_node = MDDNode(goal_node.location, curr.timestep + 1)
        new_node.updateNode(curr) 
        node_dict[(goal_node.location, curr.timestep + 1)] = new_node
        curr = new_node
        

def check_MDDs_for_conflict(node_dict1, node_dict2):
    
    
    
    

    dict1 = {} 
    dict2 = {}
    for (loc,time) in node_dict1.keys():
        
        if time in dict1: 
            dict1[time].append(loc)
        else: 
            dict1[time] = [loc]

    for (loc,time) in node_dict2.keys():
        
        if time in dict2:
            dict2[time].append(loc)
        else:
            dict2[time] = [loc]

    
    
    
    

    
    
    
    

    
    diff = abs(len(dict1) - len(dict2))
    if len(dict1) == len(dict2):
        pass
    elif len(dict1) > len(dict2): 
        
        last_timestep = max(dict2.keys())   
        for i in range(diff):               
            dict2[last_timestep+i+1] = dict2[last_timestep]  
    else:
        
        last_timestep = max(dict1.keys())
        for i in range(diff):
            dict1[last_timestep+i+1] = dict1[last_timestep]

    for time, locs in dict1.items():
        if (len(locs) == 1 and locs == dict2[time]):
            return True 

    return False

def balanceMDDs(paths1, paths2, node_dict1, node_dict2):
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

def buildJointMDD(paths1, paths2, root1, node_dict1, root2, node_dict2):
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

    
    
    
    
    q = queue.Queue()
    root = JointMDDNode(root1.location, root2.location, 0)
    q.put(root)

    
    existing_dict = {
        (root1.location, root2.location, 0): root
    }
    new_node = None
    while (not q.empty()):
    
        curr = q.get()

        (loc1,time1) = curr.location[0], curr.timestep
        (loc2,time2) = curr.location[1], curr.timestep

        node1 = node_dict1[(loc1, time1)]
        node2 = node_dict2[(loc2, time2)]

        if (len(node1.children) == 0 or len(node2.children) == 0):
            continue
        
        loc_combinations = []
        for node1_child in node1.children:
            for node2_child in node2.children:
                loc_combinations.append((node1_child.location, node2_child.location))

        
        new_node = None
        for combo in loc_combinations:
            if (combo[0] == combo[1]):
                continue
            
            params = (combo[0], combo[1], curr.timestep+1)
            if (params in existing_dict):
                
                new_node = existing_dict[params]
            else:
                new_node = JointMDDNode(combo[0], combo[1], curr.timestep+1)
                existing_dict[params] = new_node
                q.put(new_node)
            new_node.updateNode(curr)
    if (new_node == None):
        new_node = curr
    return root, new_node
            








































def check_jointMDD_for_dependency(bottom_node, paths1, paths2):
    optimal_time = max(len(paths1),len(paths2))
    goal_loc1 = paths1[-1]
    goal_loc2 = paths2[-1]
    if (bottom_node.location != [(goal_loc1),(goal_loc2)] or bottom_node.timestep != optimal_time):
        return True
    return False
