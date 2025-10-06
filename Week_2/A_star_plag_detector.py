import heapq
import numpy as np
import string




def compute_edit_distance(str1, str2):
    """To find the edit distance between two strings : str1 and str2"""
    len1, len2 = len(str1), len(str2)
    dp_matrix = np.zeros((len1 + 1, len2 + 1), dtype=int)


    for i in range(len1 + 1):
        for j in range(len2 + 1):
            if i == 0:
                dp_matrix[i][j] = j
            elif j == 0: dp_matrix[i][j] = i
            elif str1[i - 1] == str2[j - 1]: dp_matrix[i][j] = dp_matrix[i - 1][j - 1]
            else:
                dp_matrix[i][j] = 1 + min(dp_matrix[i - 1][j], dp_matrix[i][j - 1], dp_matrix[i - 1][j - 1])
    return dp_matrix[len1][len2]



def clean_text(document):
    """Tokenize and normalize the document."""
    document = document.lower().translate(str.maketrans('', '', string.punctuation))
    return document


class PuzzleState:
    def _init_(self, index1, index2, total_cost, text1, text2):
        self.index1 = index1
        self.index2 = index2
        self.total_cost = total_cost
        self.text1 = text1
        self.text2 = text2


    def _lt_(self, other):
        return (self.total_cost + self.estimate_remaining_cost()) < (other.total_cost + other.estimate_remaining_cost())


    def estimate_remaining_cost(self):
        """Estimate the remaining cost."""
        return abs(len(self.text1) - self.index1 - (len(self.text2) - self.index2))

def a_star_alignment(text1, text2):
    """Perform A* search."""
    initial_state = PuzzleState(0, 0, 0, text1, text2)
    priority_queue = []
    heapq.heappush(priority_queue, initial_state)
    visited_states = set()
    
    while priority_queue:
        current_state = heapq.heappop(priority_queue)
        if (current_state.index1, current_state.index2) in visited_states: continue
        visited_states.add((current_state.index1, current_state.index2))



        if current_state.index1 == len(text1) and current_state.index2 == len(text2):
            return current_state.total_cost



        if current_state.index1 < len(text1) and current_state.index2 < len(text2):

            cost = 0 if text1[current_state.index1] == text2[current_state.index2] else 1
            new_state = PuzzleState(current_state.index1 + 1, current_state.index2 + 1, current_state.total_cost + cost, text1, text2)
            if (new_state.index1, new_state.index2) not in visited_states:
                heapq.heappush(priority_queue, new_state)


        if current_state.index1 < len(text1):
            new_state = PuzzleState(current_state.index1 + 1, current_state.index2, current_state.total_cost + 1, text1, text2)
            if (new_state.index1, new_state.index2) not in visited_states:
                heapq.heappush(priority_queue, new_state)
        
        if current_state.index2 < len(text2):

            new_state = PuzzleState(current_state.index1, current_state.index2 + 1, current_state.total_cost + 1, text1, text2)
            if (new_state.index1, new_state.index2) not in visited_states:
                heapq.heappush(priority_queue, new_state)

    return None

def plagiarism_check(document1, document2):
    """Detect potential plagiarism by aligning two strings."""
    document1_clean = clean_text(document1)
    document2_clean = clean_text(document2)
    alignment_cost = a_star_alignment(document1_clean, document2_clean)
    
    if alignment_cost is not None:
        print(f"Potential plagiarism detected with total alignment cost: {alignment_cost}")
    else:
        print("No potential plagiarism detected.")

document1 = "I am a CSE student at IIITV. I am studying AI."
document2 = "You are studying AI. You are a CSE student at IIITV."

print("Document 1: " + document1)
print("Document 2: " + document2)
plagiarism_check(document1, document2)