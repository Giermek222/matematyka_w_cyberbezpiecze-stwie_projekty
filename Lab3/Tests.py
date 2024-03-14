import unittest
import numpy as np
from LinearIterativeTechniques.JacobiMethod import jacobi_method
from LinearIterativeTechniques.GaussSchneidelMethod import gauss_seidel_method
from LinearIterativeTechniques.Richardsonmethod import richardson_method

#Those matrices are diagonally dominant as jacobi method does not work on other kinds
equation_3x3_jordan_a = np.array([[1,0,0], [0,1,0],[0,0,1]])
equation_3x3_jordan_b = np.array([[2,1,0], [0,2,1],[1,0,2]])
equation_3x3_jordan_c = np.array([[1,-1,1], [1,3,1],[-1,0,3]])

answer_3x3 = np.array([1, 2, 3])

class Jordan_method_tests(unittest.TestCase):
    
    def test_jordan_method_for_equation_A(self):
        #given
        x0 = np.zeros_like(answer_3x3)
        solutions = jacobi_method(equation_3x3_jordan_a, answer_3x3, x0)

        #then
        np.testing.assert_array_almost_equal([1,2,3], solutions)

    def test_jordan_method_for_equation_B(self):
        #given
        x0 = np.zeros_like(answer_3x3)
        solutions = jacobi_method(equation_3x3_jordan_b, answer_3x3, x0)

        #then
        np.testing.assert_array_almost_equal([0.333333,0.333333,1.333333], solutions)

    def test_jordan_method_for_equation_C(self):
        #given
        x0 = np.zeros_like(answer_3x3)
        solutions = jacobi_method(equation_3x3_jordan_c, answer_3x3, x0)

        #then
        np.testing.assert_array_almost_equal([0.1875,0.25,1.0625], solutions)



class Gauss_method_tests(unittest.TestCase):
    
    def test_gauss_method_for_equation_A(self):
        #given
        x0 = np.zeros_like(answer_3x3)
        solutions = gauss_seidel_method(equation_3x3_jordan_a, answer_3x3, x0)

        #then
        np.testing.assert_array_almost_equal([1,2,3], solutions)

    def test_gauss_method_for_equation_B(self):
        #given
        x0 = np.zeros_like(answer_3x3)
        solutions = gauss_seidel_method(equation_3x3_jordan_b, answer_3x3, x0)

        #then
        np.testing.assert_array_almost_equal([0.333333,0.333333,1.333333], solutions)

    def test_gauss_method_for_equation_C(self):
        #given
        x0 = np.zeros_like(answer_3x3)
        solutions = gauss_seidel_method(equation_3x3_jordan_c, answer_3x3, x0)

        #then
        np.testing.assert_array_almost_equal([0.1875,0.25,1.0625], solutions)

class Richardson_method_tests(unittest.TestCase):
    
    def test_gauss_method_for_equation_A(self):
        #given
        x0 = np.zeros_like(answer_3x3)
        alpha = 0.1  # Step size parameter
        solutions = richardson_method(equation_3x3_jordan_a, answer_3x3, x0, alpha)

        #then
        np.testing.assert_array_almost_equal([1,2,3], solutions)

    def test_gauss_method_for_equation_B(self):
        #given
        x0 = np.zeros_like(answer_3x3)
        alpha = 0.1  # Step size parameter
        solutions = richardson_method(equation_3x3_jordan_b, answer_3x3, x0, alpha)

        #then
        np.testing.assert_array_almost_equal([0.333333,0.333333,1.333333], solutions)

    def test_gauss_method_for_equation_C(self):
        #given
        x0 = np.zeros_like(answer_3x3)        
        alpha = 0.1  # Step size parameter
        solutions = richardson_method(equation_3x3_jordan_c, answer_3x3, x0, alpha)

        #then
        np.testing.assert_array_almost_equal([0.1875,0.25,1.0625], solutions)

if __name__ == "__main__":
    unittest.main()