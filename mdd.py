import heapq
import queue
from single_agent_planner import build_constraint_table, build_constraint_table_with_pos, check_agent_pos_constrained, is_pos_constrained, is_constrained

class NodeDictWrap(dict):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        return super().__getitem__(item)

    def __setattr__(self, item, value):
        return super().__setitem__(item, value)

    def __lt__(self,other):
        # break ties in favor of lower h_val. 
        return self.h_val < other.h_val


def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
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
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
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
    # print(f"PUSHING heuristic: {node['g_val'] + node['h_val']} path: {get_path(node)}")

def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    # print(f"POPPING heuristic: {curr['g_val'] + curr['h_val']} path: {get_path(curr)}")
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
    closed_list = dict() #contains list of already expanded nodes
    earliest_goal_timestep = 0
    first_time = True
    paths = []
    h_value = h_values[start_loc]
    constraint_table = build_constraint_table(constraints, agent)
    pos_constraint_table = build_constraint_table_with_pos(constraints, agent)    
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'time': 0}
    push_node(open_list, root)
    closed_list[(root['loc'], root['time'])] = root
    while len(open_list) > 0:
        # print("open list to pop", open_list)
        curr = pop_node(open_list)
        # print("curr is:", get_path(curr))
        if curr['loc'] == goal_loc:
            curr_path = get_path(curr)
            # print("Current path to goal found:", curr_path)
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
                # this move violates a positive constraint from another agent
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
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['time'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['time'])] = child
                push_node(open_list, child)
            continue

        for dir in range(5):
            child_loc = move(curr['loc'], dir)
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue            
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr,
                    'time': curr['time'] + 1}

            if is_pos_constrained(curr['loc'], child['loc'], child['time'], pos_constraint_table):
                # a positive constraint on another agent means negative on this agent
                continue

            if is_constrained(curr['loc'], child['loc'], child['time'], constraint_table):
                # we dont want to do this move
                continue
            
            if (child['loc'], child['time']) in closed_list:
                existing_node = closed_list[(child['loc'], child['time'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['time'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['time'])] = child
                push_node(open_list, child)
    return paths  # Failed to find solutions

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
    
    # print("Level:", level)
    # node.display()
    # print()
    if(len(node.children) == 0):
        return
    else:
        for child in node.children:
            displayLayer(child, level+1)
   
def buildMDDTree(optimal_paths):
    root_location = optimal_paths[0][0]
    root_node = MDDNode(root_location, 0)
    # curr = root_node

    existing_nodes = {
        (root_location, 0): root_node
    }

    for path in optimal_paths:
        curr = root_node
        # for location in path[1:]:
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
        # print(f"extending {goal_node.location} once")
                
        new_node = MDDNode(goal_node.location, curr.timestep + 1)
        new_node.updateNode(curr) # add to bottom of tree
        node_dict[(goal_node.location, curr.timestep + 1)] = new_node
        curr = new_node
        # print("new node: ", new_node.display())

def check_MDDs_for_conflict(node_dict1, node_dict2):
    # given two dicts
    # indexed like dict[location] = node in mdd tree
    # build two dicts
    # indexed like dict[timestep] = [ locations ]

    dict1 = {} # timestep: [location]
    dict2 = {}
    for (loc,time) in node_dict1.keys():
        # build dict
        if time in dict1: # if this timestep was original mdd
            dict1[time].append(loc)
        else: 
            dict1[time] = [loc]

    for (loc,time) in node_dict2.keys():
        # build dict
        if time in dict2:
            dict2[time].append(loc)
        else:
            dict2[time] = [loc]

    # lst = list(dict1.keys())
    # missing_timesteps_1 = [x for x in range(lst[0], lst[-1]+1) if x not in lst] # returns the missing timesteps
    # lst = list(dict2.keys())
    # missing_timesteps_2 = [x for x in range(lst[0], lst[-1]+1) if x not in lst] # returns the missing timesteps

    # for timestep in missing_timesteps_1:
    #     dict1[timestep] = dict1[timestep-1]
    # for timestep in missing_timesteps_2:
    #     dict2[timestep] = dict2[timestep-1]

    #extend dictionary
    diff = abs(len(dict1) - len(dict2))
    if len(dict1) == len(dict2):
        pass
    elif len(dict1) > len(dict2): # mdd2/dict2 is shorter
        # extend dict2 by diff
        last_timestep = max(dict2.keys())   # last timestep = 47
        for i in range(diff):               # i = 0 1 2 3
            dict2[last_timestep+i+1] = dict2[last_timestep]  # new timesteps = 48 49 50 51
    else:
        #extend dict1
        last_timestep = max(dict1.keys())
        for i in range(diff):
            dict1[last_timestep+i+1] = dict1[last_timestep]

    for time, locs in dict1.items():
        if (len(locs) == 1 and locs == dict2[time]):
            return True #found a cardinal

    return False

def balanceMDDs(paths1, paths2, node_dict1, node_dict2):
    height1 = len(paths1[0])
    height2 = len(paths2[0])

    if (height1 != height2):
        if (height1 < height2):
            # first mdd shorter
            goal_loc = paths1[0][-1]
            bottom_node = node_dict1[(goal_loc, height1-1)]
            extendMDDTree(bottom_node, height2-height1, node_dict1)
            
        else:
            # second is shorter
            goal_loc = paths2[0][-1]
            bottom_node = node_dict2[(goal_loc, height2-1)]
            extendMDDTree(bottom_node, height1-height2, node_dict2)

def buildJointMDD(paths1, paths2, root1, node_dict1, root2, node_dict2):
    height1 = len(paths1[0])
    height2 = len(paths2[0])

    # pseudo

    # Balance both mdds:
        # if one is shorter, extend it to same length by duplicating bottom node
    if (height1 != height2):
        if (height1 < height2):
            # first mdd shorter
            goal_loc = paths1[0][-1]
            bottom_node = node_dict1[(goal_loc, height1-1)]
            extendMDDTree(bottom_node, height2-height1, node_dict1)
            
        else:
            # second is shorter
            goal_loc = paths2[0][-1]
            bottom_node = node_dict2[(goal_loc, height2-1)]
            extendMDDTree(bottom_node, height1-height2, node_dict2)

    # print(node_dict2)
    # displayLayer(node_dict2[paths2[0][0]] , 0)  
    # build root node for jointmdd
    # add it to open list
    q = queue.Queue()
    root = JointMDDNode(root1.location, root2.location, 0)
    q.put(root)

    # while open list not empty
    existing_dict = {
        (root1.location, root2.location, 0): root
    }
    new_node = None
    while (not q.empty()):
    # pop one node from open list
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

        # print("DEBUG:", loc_combinations)
        new_node = None
        for combo in loc_combinations:
            if (combo[0] == combo[1]):
                continue
            # create node if needed
            params = (combo[0], combo[1], curr.timestep+1)
            if (params in existing_dict):
                # add relationship
                new_node = existing_dict[params]
            else:
                new_node = JointMDDNode(combo[0], combo[1], curr.timestep+1)
                existing_dict[params] = new_node
                q.put(new_node)
            new_node.updateNode(curr)
    if (new_node == None):
        new_node = curr
    return root, new_node
            
#         # call helper method generateChildren(joint-node)
#             # get children of both locations from the dicts
#             # if either has no children, that means no more nodes to generate
#                 # check end condition
#             # children of the jointmdd node = compute combinations 
#                 # dont add if its same location both sides

#         # create nodes if not existing
#         # update nodes (parent/children relationship)
#         # make sure to add relationship if its an existing node already
#             # add them to an "open list"



# def main():

# #     paths1 = [[(2, 1), (2, 2), (2, 3), (2, 4), (3, 4)], [(2, 1), (2, 2), (2, 3), (3, 3), (3, 4)], [(2, 1), (2, 2), (3, 2), (3, 3), (3, 4)], [(2, 1), (3, 1), (3,
# #  2), (3, 3), (3, 4)]]
# #     paths2 = [[(1, 2), (1, 3), (2, 3), (3, 3), (4, 3)], [(1, 2), (2, 2), (2, 3), (3, 3), (4, 3)], [(1, 2), (2, 2), (3, 2), (3, 3), (4, 3)], [(1, 2), (2, 2), (3, 2), (4, 2), (4, 3)]]

#     paths1 = [[(2, 1), (2, 2), (2, 3), (2, 4), (3, 4)], [(2, 1), (2, 2), (2, 3), (3, 3), (3, 4)], [(2, 1), (2, 2), (3, 2), (3, 3), (3, 4)], [(2, 1), (3, 1), (3, 2), (3, 3), (3, 4)]]
#     paths2 = [[(1, 2), (1, 3), (1, 4)]]

#     jointMDD_root = buildJointMDD(paths1, paths2)

#     displayLayer(jointMDD_root, 0)

#     #      (2,1)
#     #    /    |
#     # (3,1)  (2,2)
#     #   |   /   |
#     # (3,2)   (2,3)
#     #  |    /   |
#     # (3,3)   (2,4)
#     #  |     /
#     #  (3,4)

# if __name__ == "__main__":
#     main()

def check_jointMDD_for_dependency(bottom_node, paths1, paths2):
    optimal_time = max(len(paths1),len(paths2))
    goal_loc1 = paths1[-1]
    goal_loc2 = paths2[-1]
    if (bottom_node.location != [(goal_loc1),(goal_loc2)] or bottom_node.timestep != optimal_time):
        return True
    return False
