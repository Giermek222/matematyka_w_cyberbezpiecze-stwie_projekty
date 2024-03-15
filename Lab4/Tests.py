import unittest
from sympy import symbols, Or, Not, And
from sympy.logic.boolalg import to_cnf, Or
from ResolutionAlgorithm import is_solvable

class TestQRDecomposition(unittest.TestCase):
    def test_resolution_algorithm_for_formula_A_returns_unsolvable(self):
        #given
        P, Q, R = symbols('P Q R')
        CNF1 = to_cnf((P | Q) & (~P | ~R) & (~Q | R))
        CNF2 = to_cnf(~P | Q)

        result = is_solvable(And(CNF1, CNF2))

        self.assertFalse(result)

    def test_resolution_algorithm_for_formula_A_returns_solvable(self):
        #given
        P, Q, R = symbols('P Q R')
        CNF1 = to_cnf((P | Q) & (~P | R) & (Q | R))
        CNF2 = to_cnf(Q)

        result = is_solvable(And(CNF1, CNF2))

        self.assertTrue(result)

    def test_resolution_algorithm_for_horn_formula_A_returns_unsolvable(self):
        #given
        P, Q, R, S = symbols('P Q R S')
        CNF1 = to_cnf(P | ~Q | ~R | ~S)
        CNF2 = to_cnf(~P | Q | ~R | ~S)
        CNF3 = to_cnf(~P | ~Q | R | ~S)
        CNF4 = to_cnf(~P | ~Q | ~R | S)

        result = is_solvable(And( And(CNF1, CNF2), And(CNF3, CNF4)))

        self.assertFalse(result)

    def test_resolution_algorithm_for_horn_formula_A_returns_solvable(self):
        #given
        P, Q, R, S = symbols('P Q R S')
        CNF1 = to_cnf(~P | ~Q | ~R | S )
        CNF2 = to_cnf(P | ~Q | ~R | ~S)

        result = is_solvable(And(CNF1, CNF2))

        self.assertTrue(result)
        



if __name__ == "__main__":
    unittest.main()