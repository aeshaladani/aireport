import numpy as np
import heapq


class MarbleSolitaireEnv:
    def __init__(self, board):
        self.board = board

    def get_start_state(self):
        return self.board

    def reached_goal(self, state):
        return np.sum(state) == 1 and state[3][3] == 1

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
            if x > 1 and state[x - 1][y] == 1 and state[x - 2][y] == 1:
                new_state = state.copy()
                new_state[x][y] = 1
                new_state[x - 2][y] = 0
                new_state[x - 1][y] = 0
                action = f"({x-2}, {y}) -> ({x}, {y})"
                new_states.append((new_state, action))

            # Moving from bottom to top
            if x < 5 and state[x + 1][y] == 1 and state[x + 2][y] == 1:
                new_state = state.copy()
                new_state[x][y] = 1
                new_state[x + 2][y] = 0
                new_state[x + 1][y] = 0
                action = f"({x+2}, {y}) -> ({x}, {y})"
                new_states.append((new_state, action))

            # Moving from left to right
            if y > 1 and state[x][y - 1] == 1 and state[x][y - 2] == 1:
                new_state = state.copy()
                new_state[x][y] = 1
                new_state[x][y - 2] = 0
                new_state[x][y - 1] = 0
                action = f"({x}, {y-2}) -> ({x}, {y})"
                new_states.append((new_state, action))

            # Moving from right to left
            if y < 5 and state[x][y + 1] == 1 and state[x][y + 2] == 1:
                new_state = state.copy()
                new_state[x][y] = 1
                new_state[x][y + 2] = 0
                new_state[x][y + 1] = 0
                action = f"({x}, {y+2}) -> ({x}, {y})"
                new_states.append((new_state, action))

        return new_states

    def count_remaining_marbles(self, state):
        return np.sum(state)

    def distance_central_marble(self, state):
        center_x, center_y = 3, 3
        distance = 0
        if state[center_x][center_y] == 1:
            return distance
        for i in range(7):
            for j in range(7):
                if state[i][j] == 1:
                    distance += abs(center_x - i) + abs(center_y - j)
        return distance


def a_star_search(env):
    priority_queue = []
    start_state = env.get_start_state()
    g_cost = 0  # Cost to reach the start state
    f_cost = (
        g_cost
        + env.count_remaining_marbles(start_state)
        + env.distance_central_marble(start_state)
    )
    heapq.heappush(
        priority_queue, (f_cost, g_cost, tuple(map(tuple, start_state)), [])
    )  # f(n), g(n)
    visited = set()

    while priority_queue:
        f_cost, g_cost, current_state, path = heapq.heappop(priority_queue)

        # Checking if the goal state is reached
        if env.reached_goal(
            np.array(current_state)
        ):  # Convert to numpy array for checking
            return np.array(current_state), path

        # Converting current_state to a tuple of tuples for immutability and hashing
        state_tuple = current_state
        if state_tuple in visited:
            continue
        visited.add(state_tuple)

        for successor, action in env.get_next_states(
            np.array(current_state)
        ):  # Convert to numpy array
            # Check if the successor has already been visited
            successor_tuple = tuple(map(tuple, successor))
            if successor_tuple in visited:
                continue

            remaining_marbles = env.count_remaining_marbles(successor)
            central_distance = env.distance_central_marble(successor)
            new_g_cost = g_cost + 1  # Increment g(n) by 1 for each move
            new_f_cost = new_g_cost + remaining_marbles + central_distance

            # Push the successor onto the priority queue
            heapq.heappush(
                priority_queue,
                (new_f_cost, new_g_cost, tuple(map(tuple, successor)), path + [action]),
            )  # Convert successor to tuple

    return None, None  # No solution found


if __name__ == "__main__":
    # Example initial state of the board (7x7)
    initial_board = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]
    )

    env = MarbleSolitaireEnv(initial_board)
    solution, path = a_star_search(env)

    if solution is not None:
        print("Solution found:")
        print(solution)
        print("Path to solution:")
        for action in path:
            print(action)
    else:
        print("No solution found.")