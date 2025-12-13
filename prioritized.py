import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost


class PrioritizedPlanningSolver(object):
    """A planner that plans for each robot sequentially."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.CPU_time = 0

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""
        num_open_cells = 0
        for row in self.my_map:
            num_open_cells += len(row)-sum(row)

        start_time = timer.time()
        result = []

        constraints = [
            # {'agent': 0,
            #  'loc': [(1,2)],
            #  'timestep': 3,
            #  'positive': True},
            # {'agent': 1,
            #  'loc': [(1,2)],
            #  'timestep': 3,
            #  'positive': True},
        ]

        for i in range(self.num_of_agents):  # Find path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, constraints)
            print(path)
            if path is None:
                raise BaseException('No solutions')
            result.append(path)

            # Add constraints here
            for time in range(len(path)):
                for agent in range(i+1, self.num_of_agents): # lower priority agents
                    vertex_constraint = {
                        'agent': agent,
                        'loc': [path[time]],
                        'timestep': time,
                        'positive': False
                    }
                    constraints.append(vertex_constraint)
                    if time > 0:
                        edge_constraint = {
                            'agent': agent,
                            'loc': [path[time], path[time-1]],
                            'timestep': time,
                            'positive': False
                        }
                        constraints.append(edge_constraint)

            # all previous agents are in the goal location and will not move. 
            # max possible path length for next agent: #open cells - #higher priority agents in goal already
            for time in range(len(path), num_open_cells - (i+1)):
                for agent in range(i+1, self.num_of_agents):
                    vertex_constraint = {
                        'agent': agent,
                        'loc': [path[-1]],
                        'timestep': time,
                        'positive': False
                    }
                    constraints.append(vertex_constraint)

        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        return result
