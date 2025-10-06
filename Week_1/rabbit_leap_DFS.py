def goal_reached(config):
    """Determine if the goal state has been reached."""
    return config == ['W', 'W', 'W', ' ', 'E', 'E', 'E']

def generate_moves(config):
    """Find all valid moves from the current configuration."""
    possible_moves = []
    empty_spot = config.index(' ')

    def swap_positions(pos1, pos2):
        """Swap elements at pos1 and pos2 and save new configuration."""
        new_config = config.copy()
        new_config[pos1], new_config[pos2] = new_config[pos2], new_config[pos1]
        possible_moves.append(new_config)

    
    if empty_spot > 0 and config[empty_spot - 1] == 'E':
        swap_positions(empty_spot, empty_spot - 1) 
    if empty_spot > 1 and config[empty_spot - 2] == 'E':
        swap_positions(empty_spot, empty_spot - 2)  

    if empty_spot < 6 and config[empty_spot + 1] == 'W':
        swap_positions(empty_spot, empty_spot + 1)  
    if empty_spot < 5 and config[empty_spot + 2] == 'W':
        swap_positions(empty_spot, empty_spot + 2)  

    return possible_moves

def depth_first_search(initial_state):
    """Perform DFS to solve the Rabbit Leap problem."""
    frontier = [(initial_state, [])]  
    explored = set() 

    while frontier:
        state, steps = frontier.pop()

        if tuple(state) in explored:
            continue
        
        explored.add(tuple(state))
        steps = steps + [state]

        if goal_reached(state):
            return steps
        
        for new_state in generate_moves(state):
            frontier.append((new_state, steps))

    return None


initial_state = ['E', 'E', 'E', ' ', 'W', 'W', 'W']


result = depth_first_search(initial_state)

if result:
    print("A solution is found:")
    for s in result:
        print(s)
else:
    print("No solution exists.")