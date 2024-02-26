import unittest
from sympy import Matrix, Rational
from Characteristic_polynomial import characteristic_polynomial
from Power_iteration import power_iteration
from Jacobi_method import jacobi_method

# Expected 1, 1, 5
matrix_A = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 5]])
# Expected -1, 1, 2
matrix_B = Matrix([[2, 1, 2], [1, 2, 0], [-2, -2, -2]])


# Expected -8, -3, -1, 0, 3
matrix_C = Matrix(
    [
        [-5, 1, -2, -4, -5],
        [3, 0, 5, -5, 1],
        [0, -1, 1, 1, 0],
        [4, -2, 4, -2, 3],
        [-3, -1, 2, 4, -3],
    ]
)


# Expected -4, -2, -1
matrix_D = Matrix([[-2,  0,  1],  [0, -2,  1],  [1,  1, -3]])

vector_A = Matrix([[1],[1],[1]])
vector_B = Matrix([[1],[3],[37]])


class TestCharacteristicPolynomial(unittest.TestCase):

    def test_characteristic_polynomial_method_for_matrix_A(self):
        [eigenvalues, eigenvectors] = characteristic_polynomial(matrix_A)
        count_1 = eigenvalues.count(1)
        count_5 = eigenvalues.count(5)
        self.assertEqual(count_1, 2)
        self.assertEqual(count_5, 1)

        self.assertIn(Matrix([[0], [0], [1]]), eigenvectors)
        self.assertIn(Matrix([[0], [1], [0]]), eigenvectors)
        self.assertIn(Matrix([[1], [0], [0]]), eigenvectors)

    def test_characteristic_polynomial_method_for_matrix_B(self):
        [eigenvalues, eigenvectors] = characteristic_polynomial(matrix_B)

        count_negative_1 = eigenvalues.count(-1)
        count_1 = eigenvalues.count(1)
        count_2 = eigenvalues.count(2)

        self.assertEqual(count_negative_1, 1)
        self.assertEqual(count_1, 1)
        self.assertEqual(count_2, 1)
        self.assertIn(Matrix([[0], [-2], [1]]), eigenvectors)
        self.assertIn(Matrix([[-1], [1], [0]]), eigenvectors)
        self.assertIn(Matrix([[-3 / 4], [1 / 4], [1]]), eigenvectors)

    def test_characteristic_polynomial_method_for_matrix_C(self):
        [eigenvalues, eigenvectors] = characteristic_polynomial(matrix_C)

        count_negative_8 = eigenvalues.count(-8)
        count_negative_3 = eigenvalues.count(-3)
        count_negative_1 = eigenvalues.count(-1)
        count_0 = eigenvalues.count(0)
        count_3 = eigenvalues.count(3)

        self.assertEqual(count_negative_8, 1)
        self.assertEqual(count_negative_3, 1)
        self.assertEqual(count_negative_1, 1)
        self.assertEqual(count_0, 1)
        self.assertEqual(count_3, 1)
        self.assertIn(
            Matrix(
                [[-1], [-Rational(1, 7)], [Rational(2, 21)], [-Rational(1, 3)], [1]]
            ),
            eigenvectors,
        )
        self.assertIn(
            Matrix(
                [[-1], [Rational(1, 5)], [Rational(3, 10)], [-Rational(1, 10)], [1]]
            ),
            eigenvectors,
        )
        self.assertIn(
            Matrix(
                [
                    [Rational(7, 15)],
                    [Rational(-109, 105)],
                    [Rational(1, 70)],
                    [Rational(-7, 6)],
                    [1],
                ]
            ),
            eigenvectors,
        )
        self.assertIn(Matrix([[-1], [-1], [0], [-1], [1]]), eigenvectors)
        self.assertIn(
            Matrix(
                [[-1], [Rational(-7, 13)], [Rational(6, 13)], [Rational(5, 13)], [1]]
            ),
            eigenvectors,
        )

class TestPowerIteration(unittest.TestCase):
    def test_power_iteration_method_for_matrix_A_vectorA(self):
        [eigenvalue, eigenvector] = power_iteration(matrix_A, vector_B)
        self.assertEqual(eigenvalue, 5)
        self.assertEqual([0, 0, 1], eigenvector)


    def test_power_iteration_method_for_matrix_B_vectorA(self):
        [eigenvalue, eigenvector] = power_iteration(matrix_B, vector_A)

        self.assertEqual(eigenvalue, 2)
        self.assertEqual([0,-2,1], eigenvector)

    def test_power_iteration_method_for_matrix_A_vectorB(self):
        [eigenvalue, eigenvector] = power_iteration(matrix_A, vector_B)
        self.assertEqual(eigenvalue, 5)
        self.assertEqual([0, 0, 1], eigenvector)


    def test_power_iteration_method_for_matrix_B_vectorB(self):
        [eigenvalue, eigenvector] = power_iteration(matrix_B, vector_B)

        self.assertEqual(eigenvalue, 2)
        self.assertEqual([0,-2,1], eigenvector)


class TestJacobian(unittest.TestCase):
    def test_jacobian_method_for_matrix_A(self):
        [eigenvalues, eigenvectors] = jacobi_method(matrix_D)
        eigenvectors_rounded = [round(elem) for elem in eigenvalues] 
        self.assertIn(-4, eigenvectors_rounded)
        self.assertIn(-2, eigenvectors_rounded)
        self.assertIn(-1, eigenvectors_rounded)

if __name__ == "__main__":
    unittest.main()
