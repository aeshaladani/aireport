
import random

# 1. Generating Uniform Random 3-SAT Problems
def generate_random_3sat(n, m):
    clauses = set()
    while len(clauses) < m:
        clause = tuple(random.sample(range(1, n + 1), 3))
        clauses.add(clause)
    return clauses

# 2. Hill-Climbing Algorithm
def hill_climbing(clauses, heuristic):
    n = max(abs(lit) for clause in clauses for lit in clause)
    solution = [random.choice([True, False]) for _ in range(n)]
    
    def evaluate(solution):
        satisfied = sum(any((lit > 0) == solution[abs(lit) - 1] for lit in clause) for clause in clauses)
        return satisfied

    for _ in range(1000):  # Max iterations
        neighbors = []
        for i in range(n):
            neighbor = solution[:]
            neighbor[i] = not neighbor[i]
            neighbors.append((evaluate(neighbor), neighbor))
        best_neighbor = max(neighbors, key=lambda x: x[0])
        if best_neighbor[0] <= evaluate(solution):
            break
        solution = best_neighbor[1]
    
    return solution, evaluate(solution)

# 3. Beam Search Algorithm
def beam_search(clauses, beam_width, heuristic):
    n = max(abs(lit) for clause in clauses for lit in clause)
    initial_population = [[random.choice([True, False]) for _ in range(n)] for _ in range(beam_width)]
    
    def evaluate(solution):
        satisfied = sum(any((lit > 0) == solution[abs(lit) - 1] for lit in clause) for clause in clauses)
        return satisfied
    
    while True:
        scored_population = [(evaluate(sol), sol) for sol in initial_population]
        scored_population.sort(reverse=True)
        best_solutions = [sol for _, sol in scored_population[:beam_width]]
        
        if max(scored_population)[0] == len(clauses):
            break
        
        initial_population = []
        for sol in best_solutions:
            for i in range(n):
                neighbor = sol[:]
                neighbor[i] = not neighbor[i]
                initial_population.append(neighbor)
        
        if len(initial_population) <= beam_width:
            break
            
    return best_solutions[0], evaluate(best_solutions[0])

# 4. Variable Neighborhood Descent
def variable_neighborhood_descent(clauses, neighborhoods):
    n = max(abs(lit) for clause in clauses for lit in clause)
    solution = [random.choice([True, False]) for _ in range(n)]
    
    def evaluate(solution):
        satisfied = sum(any((lit > 0) == solution[abs(lit) - 1] for lit in clause) for clause in clauses)
        return satisfied

    while True:
        improvement = False
        for neighborhood in neighborhoods:
            neighbors = []
            for i in range(n):
                neighbor = solution[:]
                neighbor[i] = not neighbor[i]
                neighbors.append((evaluate(neighbor), neighbor))
            best_neighbor = max(neighbors, key=lambda x: x[0])
            if best_neighbor[0] > evaluate(solution):
                solution = best_neighbor[1]
                improvement = True
                break
        if not improvement:
            break
    
    return solution, evaluate(solution)

# 5. Heuristic Functions
def heuristic_one(solution):
    return sum(1 for lit in solution if lit)

def heuristic_two(solution):
    return sum(1 for lit in solution if not lit)

# 6. Running Experiments
def run_experiments(n_values, m_values):
    results = []
    for n in n_values:
        for m in m_values:
            clauses = generate_random_3sat(n, m)
            # Hill Climbing
            hill_climb_result = hill_climbing(clauses, heuristic_one)
            # Beam Search
            beam_search_result_3 = beam_search(clauses, 3, heuristic_one)
            beam_search_result_4 = beam_search(clauses, 4, heuristic_one)
            # Variable Neighborhood Descent
            neighborhoods = [lambda x: x] * 3  # Placeholder for actual neighborhood functions
            vnd_result = variable_neighborhood_descent(clauses, neighborhoods)

            results.append({
                "n": n,
                "m": m,
                "hill_climb": hill_climb_result,
                "beam_search_3": beam_search_result_3,
                "beam_search_4": beam_search_result_4,
                "vnd": vnd_result
            })
    return results

# Defining values for n (number of variables) and m (number of clauses)
n_values = [10, 20, 30]  # Example variable counts
m_values = [20, 40, 60]  # Example clause counts

results = run_experiments(n_values, m_values)

for result in results:
    print(f"3-SAT Instance with n={result['n']} m={result['m']}:")
    print(f"  Hill Climbing Result: {result['hill_climb']}")
    print(f"  Beam Search (3) Result: {result['beam_search_3']}")
    print(f"  Beam Search (4) Result: {result['beam_search_4']}")
    print(f"  Variable Neighborhood Descent Result: {result['vnd']}")
    print("\n")
