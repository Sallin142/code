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

        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        result = []
        constraints = []

        for i in range(self.num_of_agents):
            base_distance = self.heuristics[i][self.starts[i]]
            higher_priority_path_lengths = sum(len(result[j]) - 1 for j in range(i))
            buffer = (len(self.my_map) + len(self.my_map[0])) * 5
            max_timestep = base_distance + higher_priority_path_lengths + buffer
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, constraints, max_timestep)
            if path is None:
                raise BaseException('No solutions')
            result.append(path)

            for t in range(len(path)):
                for j in range(i + 1, self.num_of_agents):
                    constraints.append({'agent': j, 'loc': [path[t]], 'timestep': t})
            for t in range(len(path) - 1):
                for j in range(i + 1, self.num_of_agents):
                    constraints.append({'agent': j, 'loc': [path[t+1], path[t]], 'timestep': t+1})
            goal_location = path[-1]
            goal_timestep = len(path) - 1
            max_horizon = len(self.my_map) * len(self.my_map[0])       
            for t in range(goal_timestep + 1, goal_timestep + max_horizon):
                for j in range(i + 1, self.num_of_agents):
                    constraints.append({'agent': j, 'loc': [goal_location], 'timestep': t, 'final': True})    

        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        return result
