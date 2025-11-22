import random
import copy

# Board cell values: ' ' (empty), 'X', 'O'
# MENACE will play 'X' and always move first in this simplified setup.

def board_to_key(board):
    return ''.join(board)

def available_moves(board):
    return [i for i, v in enumerate(board) if v == ' ']

def check_winner(board):
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in wins:
        if board[a]==board[b]==board[c] and board[a] != ' ':
            return board[a]
    if ' ' not in board:
        return 'D'  # draw
    return None

class MENACE:
    def __init__(self, init_beads=3):
        # map state_key -> dict(move_index -> bead_count)
        self.boxes = {}
        self.init_beads = init_beads

    def ensure_box(self, board):
        key = board_to_key(board)
        if key in self.boxes:
            return
        moves = available_moves(board)
        # initialize beads uniformly for each legal move
        self.boxes[key] = {m: self.init_beads for m in moves}

    def choose_move(self, board):
        self.ensure_box(board)
        key = board_to_key(board)
        beads = self.boxes[key]
        # probabilistic choice proportional to bead counts
        total = sum(beads.values())
        r = random.randint(1, total)
        cum = 0
        for move, count in beads.items():
            cum += count
            if r <= cum:
                return move

    def update_history(self, history, outcome, add_beads=2, remove_beads=1):
        # history: list of (state_key, move)
        if outcome == 'X':  # MENACE won -> reinforce
            for key, move in history:
                self.boxes[key][move] = self.boxes[key].get(move, 0) + add_beads
        elif outcome == 'O':  # MENACE lost -> penalize
            for key, move in history:
                # remove beads but keep at least 1 bead so move remains possible
                self.boxes[key][move] = max(1, self.boxes[key].get(move, 1) - remove_beads)
        # draw: do nothing

def random_opponent_move(board):
    moves = available_moves(board)
    return random.choice(moves)

def play_game(agent, verbose=False):
    board = [' '] * 9
    history = []
    turn = 'X'  # MENACE starts
    while True:
        if turn == 'X':
            move = agent.choose_move(board)
            history.append((board_to_key(board), move))
            board[move] = 'X'
        else:
            move = random_opponent_move(board)
            board[move] = 'O'
        winner = check_winner(board)
        if winner is not None:
            agent.update_history(history, winner)
            return winner  # 'X', 'O', or 'D'
        turn = 'O' if turn == 'X' else 'X'

def train_menace(n_games=5000, seed=0):
    random.seed(seed)
    agent = MENACE(init_beads=3)
    results = {'X':0, 'O':0, 'D':0}
    for i in range(n_games):
        w = play_game(agent)
        results[w] += 1
        if (i+1) % (n_games//5) == 0:
            print(f"After {i+1} games: {results}")
    return agent, results

if __name__ == '__main__':
    agent, results = train_menace(5000, seed=42)
    print("Final results:", results)
    # Optionally inspect a few boxes
    sample_keys = list(agent.boxes.keys())[:5]
    for k in sample_keys:
        print("state:", k, "beads:", agent.boxes[k])
