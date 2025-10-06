from collections import deque

def goal_reached(state):
    """Returns True if the current state is the goal."""
    return state == ['W', 'W', 'W', ' ', 'E', 'E', 'E']

def generate_moves(state):
    """Generate all possible valid moves from the current state."""
    
    possible_moves = []
    space_index = state.index(' ') 
    
    
    def move_and_record(i, j):
        new_state = state[:]
        new_state[i], new_state[j] = new_state[j], new_state[i]
        possible_moves.append(new_state)
    
   
    if space_index - 1 >= 0 and state[space_index - 1] == 'E':
        move_and_record(space_index, space_index - 1)
    if space_index - 2 >= 0 and state[space_index - 2] == 'E':
        move_and_record(space_index, space_index - 2)
    
    
    if space_index + 1 < len(state) and state[space_index + 1] == 'W':
        move_and_record(space_index, space_index + 1)
    if space_index + 2 < len(state) and state[space_index + 2] == 'W':
        move_and_record(space_index, space_index + 2)
    
    return possible_moves

def bfs_solver(start):
    """Performs BFS to find the path from start to the goal state."""
    to_visit = deque([(start, [])])  
    explored = set()
    
    while to_visit:
        state, path_so_far = to_visit.popleft()
        
        state_tuple = tuple(state)
        if state_tuple in explored:
            continue
        
        explored.add(state_tuple)
        path_so_far = path_so_far + [state]  
        
        if goal_reached(state):
            return path_so_far  
        
        for next_state in generate_moves(state):
            to_visit.append((next_state, path_so_far))
    
    return None  


initial_state = ['E', 'E', 'E', ' ', 'W', 'W', 'W']

result = bfs_solver(initial_state)

if result:
    print("Steps to reach the goal:")
    for step in result:
        print(step)
else:
    print("No solution exists.")