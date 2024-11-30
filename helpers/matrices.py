from typing import List, Tuple, Callable
from vectors import dot, Vector

Matrix = List[List[float]]

def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A (as a Vector)"""
    return A[i]

def get_column(A: Matrix, j: int) -> Vector:
    """Returns the j-th column of A (as a Vector)"""
    return [A_i[j] for A_i in A]

def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    """Returns a num_rows x num_cols matrix whose (i, j)-th entry is entry_fn(i, j)"""
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]

def identity_matrix(n: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

def matrix_add(A: Matrix, B: Matrix) -> Matrix:
    """Adds corresponding elements of two matrices"""
    num_rows, num_cols = shape(A)
    return [[A[i][j] + B[i][j] for j in range(num_cols)] for i in range(num_rows)]

def matrix_subtract(A: Matrix, B: Matrix) -> Matrix:
    """Subtracts corresponding elements of two matrices"""
    num_rows, num_cols = shape(A)
    return [[A[i][j] - B[i][j] for j in range(num_cols)] for i in range(num_rows)]

def matrix_scalar_multiply(c: float, A: Matrix) -> Matrix:
    """Multiplies every element of a matrix by c"""
    num_rows, num_cols = shape(A)
    return [[c * A[i][j] for j in range(num_cols)] for i in range(num_rows)]

def matrix_vector_multiply(A: Matrix, v: Vector) -> Vector:
    """Multiplies an A x b matrix by a b x 1 vector"""
    b = len(v)
    return [dot(get_row(A, i), v) for i in range(b)]