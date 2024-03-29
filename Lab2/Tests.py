import unittest
import numpy as np
from DecompositionMethods.DiagonalDecomposition import diagonal_decomposition
from DecompositionMethods.CholeskyDecomposition import cholesky_decomposition
from DecompositionMethods.LUDecomposition import lu_decomposition
from DecompositionMethods.QRDecomposition import qr_decomposition
from DecompositionMethods.JordanDecomposition import jordan_decomposition

matrix_A = np.array([[2, 1, 2], [1, 2, 0], [-2, -2, -2]])
matrix_B = np.array([[-2,  0,  1],  [0, -2,  1],  [1,  1, -3]])
matrix_C = np.array([[1,2,3,2],[1,3,2,1],[5,0,1,3]])
matrix_D = np.array([[1,2],[2,1],[5,3]])
matrix_symetric_positive_definite = np.array([[4, 2, -2],[2, 10, 4],[-2, 4, 8]])


class MatrixDiagonalization(unittest.TestCase):
    def test_matrix_diagonalisation_for_matrix_A(self):
        #given
        D, P = diagonal_decomposition(matrix_A)

        #then
        P_inv = np.linalg.inv(P)
        result = np.dot(np.dot(P, D), P_inv)
        np.testing.assert_array_almost_equal(matrix_A, result)

    def test_matrix_diagonalisation_for_matrix_B(self):       
        #given
        D, P = diagonal_decomposition(matrix_B)

        #then
        P_inv = np.linalg.inv(P)
        result = np.dot(np.dot(P, D), P_inv)
        np.testing.assert_array_almost_equal(matrix_B, result)


class TestJordansDecomposition(unittest.TestCase):
    def test_jordan_decomposition_for_matrix_A(self):
        #given
        D, P = jordan_decomposition(matrix_A)

        #then
        P_inv = np.linalg.inv(P)
        result = np.dot(np.dot(P, D), P_inv)
        np.testing.assert_array_almost_equal(matrix_A, result)

    def test_jordan_decomposition_for_matrix_B(self):
        #given
        D, P = jordan_decomposition(matrix_B)

        #then
        P_inv = np.linalg.inv(P)
        result = np.dot(np.dot(P, D), P_inv)
        np.testing.assert_array_almost_equal(matrix_B, result)


class TestLUDecompositon(unittest.TestCase):
    def test_lu_decomposition_for_matrix_A(self):
        #given
        L, U = lu_decomposition(matrix_A)

        #then
        result = np.dot(L, U)
        np.testing.assert_array_almost_equal(matrix_A, result)

    def test_lu_decomposition_for_matrix_B(self):
        #given
        L, U = lu_decomposition(matrix_B)

        #then
        result = np.dot(L, U)
        np.testing.assert_array_almost_equal(matrix_B, result)

    def test_lu_decomposition_for_matrix_C(self):
        #given
        L, U = lu_decomposition(matrix_C)

        #then
        result = np.dot(L, U)
        np.testing.assert_array_almost_equal(matrix_C, result)

    def test_lu_decomposition_for_matrix_D(self):
        #given
        L, U = lu_decomposition(matrix_D)

        #then
        self.assertIsNone(L)
        self.assertIsNone(U)


class TestCholeskyDecomposition(unittest.TestCase):
    def test_cholesky_decomposition_for_matrix_symetric_positive_definite(self):
        L = cholesky_decomposition(matrix_symetric_positive_definite)
        
        result = np.dot(L, L.T)
        np.testing.assert_array_almost_equal(matrix_symetric_positive_definite, result)


class TestQRDecomposition(unittest.TestCase):
    def test_qr_decomposition_for_matrix_A(self):
        #given
        Q, R = qr_decomposition(matrix_A)

        #then
        result = np.dot(Q, R)
        np.testing.assert_array_almost_equal(matrix_A, result)

    def test_qr_decomposition_for_matrix_B(self):
        #given        
        Q, R = qr_decomposition(matrix_B)

        #then
        result = np.dot(Q, R)
        np.testing.assert_array_almost_equal(matrix_B, result)

if __name__ == "__main__":
    unittest.main()