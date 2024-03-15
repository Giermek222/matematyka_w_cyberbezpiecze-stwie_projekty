from sympy import symbols, Or, Not, And
from sympy.logic.boolalg import to_cnf, Or

def resolution(CNF):
    clauses = list(CNF.args)

    while True:
        new_clauses = set()

        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                resolvents = resolve(clauses[i], clauses[j])

                if False in resolvents:
                    return True  # formula is unsolvable

                new_clauses.update(resolvents)

        if new_clauses.issubset(clauses):
            return False  # formula is solvable if it has any elements left after resolution

        clauses.extend(new_clauses)

def resolve(ci, cj):
    resolvents = []

    for di in ci.args:
        for dj in cj.args:
            if di == ~dj or ~di == dj:
                resolvent = Or(*[clause for clause in (ci.args + cj.args) if clause != di and clause != dj])
                resolvents.append(resolvent)

    return resolvents

# Example usage
if __name__ == "__main__":
    P, Q, R = symbols('P Q R')

    CNF1 = to_cnf((P | Q) & (~P | ~R) & (~Q | R))
    CNF2 = to_cnf(~P | Q)

   
    result = resolution(And(CNF1, CNF2))
    print("Contradiction Found!" if result else "No contradiction found.")