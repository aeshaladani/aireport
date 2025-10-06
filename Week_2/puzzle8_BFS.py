
from collections import deque
import random



class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

def get_successors(node):
    successors = []
    index = node.state.index(0)  # Find the index of the blank (0)
    quotient = index // 3
    remainder = index % 3

    # Row constrained moves
    if quotient == 0:
        moves = [3]  # Move down only
    if quotient == 1:
        moves = [-3, 3]  # Move up or down
    if quotient == 2:
        moves = [-3]  # Move up only

    # Column constrained moves
    if remainder == 0:
        moves += [1]  # Move right only
    if remainder == 1:
        moves += [-1, 1]  # Move left or right
    if remainder == 2:
        moves += [-1]  # Move left only

    for move in moves:
        new_index = index + move
        if 0 <= new_index < 9:  # Ensure it's within the puzzle bounds
            new_state = list(node.state)
            # Swap the blank with the adjacent number
            new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
            successor = Node(new_state, node)
            successors.append(successor)
            
    return successors

def bfs(start_state, goal_state):
    start_node = Node(start_state)
    goal_node = Node(goal_state)
    queue = deque([start_node])
    visited = set()
    nodes_explored = 0
    
    while queue:
        node = queue.popleft()
        
        if tuple(node.state) in visited:
            continue
        
        visited.add(tuple(node.state))
        nodes_explored += 1
        
        if node.state == goal_node.state:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print('Total nodes explored:', nodes_explored)
            return path[::-1]
        
        for successor in get_successors(node):
            queue.append(successor)
    
    return None

# Initial state of the puzzle
start_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]  # Solved puzzle

# Generate a deeper goal state by applying 20 random moves
s_node = Node(start_state)
D = 20  # Depth of the goal state
d = 0

while d < D:
    # Randomly select one of the successors to generate the goal state
    successors = get_successors(s_node)
    s_node = random.choice(successors)
    d += 1

goal_state = s_node.state  # The randomly generated goal state

# Run BFS to find the solution
solution = bfs(start_state, goal_state)

if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")
