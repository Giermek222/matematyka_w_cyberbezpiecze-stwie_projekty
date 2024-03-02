import unittest
import numpy as np
from DecompositionMethods.DiagonalDecomposition import diagonal_decomposition
from DecompositionMethods.CholeskyDecomposition import cholesky_decomposition

matrix_A = np.array([[2, 1, 2], [1, 2, 0], [-2, -2, -2]])
matrix_B = np.array([[-2,  0,  1],  [0, -2,  1],  [1,  1, -3]])
matrix_symetric_positive_definite = np.array([[4, 2, -2],
              [2, 10, 4],
              [-2, 4, 8]])


class MatrixDiagonalization(unittest.TestCase):
    def test_matrix_diagonalisation_for_matrix_A(self):
        #given
        D, P = diagonal_decomposition(matrix_A)

        #then
        P_inv = np.linalg.inv(P)
        result = np.dot(np.dot(P, D), P_inv)
        result = np.round(result,3)
        np.testing.assert_array_almost_equal(matrix_A, result)

    def test_matrix_diagonalisation_for_matrix_B(self):       
        #given
        D, P = diagonal_decomposition(matrix_B)

        #then
        P_inv = np.linalg.inv(P)
        result = np.dot(np.dot(P, D), P_inv)
        result = np.round(result,3)
        np.testing.assert_array_almost_equal(matrix_B, result)


class TestJordansDecomposition(unittest.TestCase):
    def chekckForMatrixA(self):
        self.assertEquals(1, 1)

    def chekckForMatrixb(self):
        self.assertEquals(1, 1)


class TestLUDecompositon(unittest.TestCase):
    def chekckForMatrixA(self):
        self.assertEquals(1, 1)

    def chekckForMatrixb(self):
        self.assertEquals(1, 1)


class TestCholeskyDecomposition(unittest.TestCase):
    def test_cholesky_decomposition_for_matrix_symetric_positive_definite(self):
        L = cholesky_decomposition(matrix_symetric_positive_definite)
        
        result = np.dot(L, L.T)
        np.testing.assert_array_almost_equal(matrix_symetric_positive_definite, result)


class TestQRDecomposition(unittest.TestCase):
    def chekckForMatrixA(self):
        self.assertEquals(1, 1)

    def chekckForMatrixb(self):
        self.assertEquals(1, 1)

if __name__ == "__main__":
    unittest.main()