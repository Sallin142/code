import heapq

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]

def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst

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

def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  

def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path

def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    
    
    

    if next_time not in constraint_table:
        return False

    constraints = constraint_table[next_time]
    for constraint in constraints:
        location = constraint['loc']
        is_positive = constraint['positive']
        if len(location) == 1:
            
            if not is_positive and location[0] == next_loc:
                return True  
        else:
            
            if not is_positive and location[0] == curr_loc and location[1] == next_loc:
                return True

    return False

def is_pos_constrained(curr_loc, next_loc, next_time, pos_constraint_table):
    
    if next_time not in pos_constraint_table:
        return False

    constraints = pos_constraint_table[next_time]
    for constraint in constraints:
        location = constraint['loc']
        if len(location) == 1:
            
            if location[0] == next_loc:
                return True
        else:
            
            if location[0] == next_loc and location[1] == curr_loc:
                return True

    return False

def check_agent_pos_constrained(curr, constraint_table):
    
    next_time = curr['time'] + 1
    if next_time not in constraint_table:
        return None
    constraints = constraint_table[next_time]
    for constraint in constraints:
        if not constraint['positive']:
            continue
        
        location = constraint['loc']
        if len(location) == 1:
            return location[0]
        else:
            if curr['loc'] == location[0]:
                return location[1]
            else:
                
                return ()
    return None

def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))

def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr

def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

def goal_constraint_exists(goal_loc, time, constraint_table, pos_constraint_table):
    """The agent is in the goal, return True if there is a future constraint on this goal location"""
    timestep_constraints = [v for k,v in constraint_table.items() if k > time]
    timestep_pos_constraints = [v for k,v in pos_constraint_table.items() if k >= time]
    
    for constraints in timestep_constraints:
        for constraint in constraints:
            if len(constraint['loc']) == 1:
                
                if constraint['loc'][0] == goal_loc and not constraint['positive']: 
                    return True
                if constraint['loc'][0] != goal_loc and constraint['positive']: 
                    return True
            else:
                
                if constraint['positive']:
                    return True

    for pos_constraints in timestep_pos_constraints:
        for pos_constraint in pos_constraints:
            if len(pos_constraint['loc']) == 1:
                if pos_constraint['loc'][0] == goal_loc:
                    return True
            else:
                if pos_constraint['loc'][1] == goal_loc:
                    return True
    return False

def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """
    num_open_cells = 0 
    for row in my_map:
        num_open_cells += len(row)-sum(row)

    
    open_list = []
    closed_list = dict() 
    earliest_goal_timestep = 0
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
        if curr['loc'] == goal_loc and not goal_constraint_exists(goal_loc, curr['time'], constraint_table, pos_constraint_table):
            return get_path(curr)

        next_loc = check_agent_pos_constrained(curr, constraint_table)
        if next_loc == ():
            continue

        if next_loc != None:
            if is_pos_constrained(curr['loc'], next_loc, curr['time'] + 1, pos_constraint_table):
                
                continue

            if next_loc[0] < 0 or next_loc[0] >= len(my_map)                or next_loc[1] < 0 or next_loc[1] >= len(my_map[0]):
                return None
            if my_map[next_loc[0]][next_loc[1]]:
                return None

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

    return None  
