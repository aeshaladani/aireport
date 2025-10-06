import random


def generate_k_sat(k, m, n):

    """
    k (int): Number of literals per clause.
    m (int): Number of clauses.
    n (int): Number of variables.
    """

    clauses = []
    variables = list(range(1, n + 1))

    for _ in range(m):
        clause = random.sample(variables, k)  # Selecting k distinct variables
        # Negating some literals randomly
        clause = [(var if random.choice([True, False]) else -var) for var in clause]
        clauses.append(clause)

    return clauses


def main():
    print("K-SAT Problem Generator")

    # Get input from the user
    k = int(input("Enter the number of literals per clause (k): "))
    m = int(input("Enter the number of clauses (m): "))
    n = int(input("Enter the number of variables (n): "))

    # Generate the k-SAT problem
    clauses = generate_k_sat(k, m, n)

    # Display the generated clauses
    print("\nGenerated Clauses:")
    for i, clause in enumerate(clauses, 1):
        print(f"Clause {i}: {clause}")


if __name__ == "__main__":
    main()