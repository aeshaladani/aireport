import numpy as np
import heapq


class Environment:

    def __init__(self, start_state=None, goal_state=None):
        self.actions = [1, 2, 3, 4]  # 1 - Up, 2 - Down, 3 - Right, 4 - Left
        if goal_state is None:
            self.goal_state = self.generate_goal_state()
        else:
            self.goal_state = goal_state
        if start_state is None:
            self.start_state = self.generate_start_state()
        else:
            self.start_state = start_state

    def generate_start_state(self):
        start = np.zeros((7, 7))
        x = (0, 1, 5, 6)
        y = (0, 1, 5, 6)

        for i in x:
            for j in y:
                start[i][j] = -1

        x = (2, 3, 4)
        y = range(7)

        for i in x:
            for j in y:
                start[i][j] = 1
                start[j][i] = 1
        start[3][3] = 0

        return start

    def generate_goal_state(self):
        goal = np.zeros((7, 7))
        x = (0, 1, 5, 6)
        y = (0, 1, 5, 6)

        for i in x:
            for j in y:
                goal[i][j] = -1

        x = (2, 3, 4)
        y = range(7)

        for i in x:
            for j in y:
                goal[i][j] = 0
                goal[j][i] = 0
        goal[3][3] = 1
        return goal

    def get_start_state(self):
        return self.start_state

    def get_goal_state(self):
        return self.goal_state

    def get_next_states(self, state):
        new_states = []
        spaces = []
        for i in range(7):
            for j in range(7):
                if state[i][j] == 0:
                    spaces.append((i, j))

        for space in spaces:
            x, y = space

            # Moving from top to bottom
            if x > 1:
                if state[x - 1][y] == 1 and state[x - 2][y] == 1:
                    new_state = state.copy()
                    new_state[x][y] = 1
                    new_state[x - 2][y] = 0
                    new_state[x - 1][y] = 0
                    action = f"({x - 2}, {y}) -> ({x}, {y})"
                    new_states.append((new_state, action))

            # Moving from bottom to top
            if x < 5:
                if state[x + 1][y] == 1 and state[x + 2][y] == 1:
                    new_state = state.copy()
                    new_state[x][y] = 1
                    new_state[x + 2][y] = 0
                    new_state[x + 1][y] = 0
                    action = f"({x + 2}, {y}) -> ({x}, {y})"
                    new_states.append((new_state, action))

            # Moving from left to right
            if y > 1:
                if state[x][y - 1] == 1 and state[x][y - 2] == 1:
                    new_state = state.copy()
                    new_state[x][y] = 1
                    new_state[x][y - 2] = 0
                    new_state[x][y - 1] = 0
                    action = f"({x}, {y - 2}) -> ({x}, {y})"
                    new_states.append((new_state, action))

            if y < 5:
                if state[x][y + 1] == 1 and state[x][y + 2] == 1:
                    new_state = state.copy()
                    new_state[x][y] = 1
                    new_state[x][y + 2] = 0
                    new_state[x][y + 1] = 0
                    action = f"({x}, {y + 2}) -> ({x}, {y})"
                    new_states.append((new_state, action))

        return new_states

    def reached_goal(self, state):
        return np.array_equal(state, self.goal_state)


def solve_marble_solitaire(env):
    priority_queue = []
    heapq.heappush(
        priority_queue, (0, env.get_start_state(), [])
    )  # (cost, state, path)
    visited = set()

    while priority_queue:
        cost, current_state, path = heapq.heappop(priority_queue)

        # Checking if the goal state is reached
        if env.reached_goal(current_state):
            return current_state, path

        # Converting current_state to tuple for immutability and hashing
        state_tuple = tuple(
            map(tuple, current_state)
        )  # Converting to a tuple of tuples for set operations
        if state_tuple in visited:
            continue
        visited.add(state_tuple)

        # Converting state_tuple back to a NumPy array for processing
        current_state_array = np.array(current_state)

        for successor, action in env.get_next_states(current_state_array):
            new_cost = cost + 1  # Incrementing the path cost for each move
            new_path = path + [action]  # Appending the action to the path

            # Converting successor to tuple for hashing
            successor_tuple = tuple(map(tuple, successor))
            heapq.heappush(
                priority_queue, (new_cost, successor_tuple, new_path)
            )  # Pushing with current cost and path

    return None, None  # No solution found


# Instantiating the environment and solving
env = Environment()
solution, path = solve_marble_solitaire(env)

# Printing the final solution and the path taken to reach it
if solution is not None:
    print("Solution (Final Board Configuration):")
    for row in solution:
        print(row)
    print("\nPath to Solution (List of Moves):")
    for action in path:
        print(action)
else:
    print("No solution found.")