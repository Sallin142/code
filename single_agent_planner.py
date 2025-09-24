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


def build_constraint_table(constraints, agent):
    table = dict()
    for constraint in constraints:
        if constraint['agent'] == agent:
            if constraint['timestep'] not in table:
                table[constraint['timestep']] = [constraint]
            else:
                table[constraint['timestep']].append(constraint)
        
        if constraint['agent'] != agent and constraint.get('positive', False) == True:
            if len(constraint['loc']) > 1:
                cons_i = {'agent': agent,
                            'loc': [constraint['loc'][1], constraint['loc'][0]],
                            'timestep': constraint['timestep'],
                            'positive': False
                            }
            else:
                cons_i = {'agent': agent,
                        'loc': constraint['loc'],
                        'timestep': constraint['timestep'],
                        'positive': False
                        }
            if cons_i['timestep'] not in table:
                table[cons_i['timestep']] = [cons_i]
            else:
                table[cons_i['timestep']].append(cons_i)

    return table


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table, mode="tri"):

    if mode not in ("tri", "bool"):
        raise ValueError("mode must be 'tri' or 'bool'")

    if next_time in constraint_table:
        constraints = constraint_table[next_time]
        for c in constraints:
            is_vertex_match = (len(c['loc']) == 1 and c['loc'] == [next_loc])
            is_edge_match = (len(c['loc']) != 1 and c['loc'] == [curr_loc, next_loc])

            if is_vertex_match or is_edge_match:
                if mode == "tri":
                    return 1 if c.get('positive', False) else 0
                else:
                    return 1

    constraints_before = [
        c
        for t, clist in constraint_table.items()
        if t < next_time
        for c in clist
    ]

    for c in constraints_before:
        if c.get('final', False) and c['loc'] == [next_loc]:
            if mode == "tri":
                return 0
            else:
                return 1

    if mode == "tri":
        return -1
    else:
        return 0



def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints,
                    max_timestep=None):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
        max_timestep - maximum timestep to search (None for unlimited)
    """
    has_positive_constraints = any(
        constraint.get('positive', False) == True 
        for constraint in constraints
    )
    
    mode = "tri" if has_positive_constraints else "bool"

    open_list = []
    closed_list = {}
    earliest_goal_timestep = 0
    h_value = h_values[start_loc]
    constraint_table = build_constraint_table(constraints, agent)

    root = {
        'loc': start_loc,
        'g_val': 0,
        'h_val': h_values[start_loc],
        'parent': None,
        'time': 0
    }

    push_node(open_list, root)
    closed_list[(root['loc'])] = root
    while len(open_list) > 0:
        curr = pop_node(open_list)

        future_constraints = [
            c
            for t, clist in constraint_table.items()
            if t > curr['time']
            for c in clist
        ]

        if curr['loc'] == goal_loc:
            if mode == "tri":
                no_future_block = True
                for c in future_constraints:
                    if c['loc'] == [goal_loc] and (not c.get('positive', False)):
                        no_future_block = False
                        break
                if no_future_block:
                    return get_path(curr)

            else:
                goal_constrained = any(c['loc'] == [goal_loc] for c in future_constraints)
                if not goal_constrained:
                    return get_path(curr)

        if mode == "tri":
            forced_child_added = False

            for d in range(5):
                child_loc = move(curr['loc'], d)
                child_time = curr['time'] + 1

                status = is_constrained(curr['loc'], child_loc, child_time,
                                        constraint_table, 'tri')

                if status == 1:
                    r, c = child_loc
                    if r < 0 or r >= len(my_map):
                        continue
                    if c < 0 or c >= len(my_map[r]):
                        continue
                    if my_map[r][c]:
                        continue

                    if max_timestep is not None and child_time > max_timestep:
                        continue

                    child = {
                        'loc': child_loc,
                        'g_val': curr['g_val'] + 1,
                        'h_val': h_values[child_loc],
                        'parent': curr,
                        'time': child_time
                    }

                    key = (child['loc'], child['time'])
                    if key in closed_list:
                        existing_node = closed_list[key]
                        if compare_nodes(child, existing_node):
                            closed_list[key] = child
                            push_node(open_list, child)
                    else:
                        closed_list[key] = child
                        push_node(open_list, child)

                    forced_child_added = True
                    break

            if forced_child_added:
                continue

        for d in range(5): 
            child_loc = move(curr['loc'], d)
            child_time = curr['time'] + 1

            
            r, c = child_loc
            if r < 0 or r >= len(my_map):
                continue
            if c < 0 or c >= len(my_map[r]):
                continue
            if my_map[r][c]:
                continue

          
            if max_timestep is not None and child_time > max_timestep:
                continue

           
            status = is_constrained(curr['loc'], child_loc, child_time,
                                    constraint_table, mode)

           
            if (mode == "tri" and status == 0) or (mode == "bool" and status == 1):
                continue

            child = {
                'loc': child_loc,
                'g_val': curr['g_val'] + 1,
                'h_val': h_values[child_loc],
                'parent': curr,
                'time': child_time
            }

            key = (child['loc'], child['time'])
            if key in closed_list:
                existing_node = closed_list[key]
                if compare_nodes(child, existing_node):
                    closed_list[key] = child
                    push_node(open_list, child)
            else:
                closed_list[key] = child
                push_node(open_list, child)

    return None  # Failed to find solutions
