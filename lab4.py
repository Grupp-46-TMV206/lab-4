import numpy as np
from scipy import linalg as LA
from sympy import Matrix
import time


# a)

A = Matrix([[1, 2, 1, -1, 2, 0], 
            [3, 4, 5, 2, 0, 0], 
            [2, 2, 1, 0, 2, 0]])
X = A.rref()
# print(X) Gives us:
"""
x1 + x4 = 0
x2 - 3x4/2 + 5x5/3 = 0
x3 + x4 - 4x5/3 = 0

To solve it we introduce variables s and t.
They are real numbers.
x1 = s
x2 = -3s/2 - 5t/3
x3 = s + 4t/3
x4 = -s
x5 = t
"""


# b)

def verify_solution(A, x, y):
    # Return True if very close to solution (error )
    if sum(np.matmul(A, x) - np.array(y)) < 0.1:
        print("Correct solution!\n")
    else:
        print("Incorrect solution!\n")

def gauss_solve(total_matrix, matrix_method):

    # Every column but the last is A
    A = total_matrix[:, :-1]
    # Last column represents y
    y = total_matrix[:, -1]

    start_time = time.time()
    # Use gauss elimination to get the solution
    # in the last column
    row_reduced = total_matrix.rref()
    end_time = time.time()
    print(f"It took {end_time - start_time} seconds using {matrix_method} and the gauss solution!")

    x = row_reduced[0][:, -1] # Last column is the solution
    verify_solution(A, x, y)


# Test with np.random.rand
total_matrix = Matrix(np.random.rand(100, 101)) # 101th column is the y-vector
gauss_solve(total_matrix, "np.random.rand")

# Test with np.random.randint
total_matrix = Matrix([[np.random.randint(0, 1_000_000_000) for _ in range(101)] for _ in range(100)])
gauss_solve(total_matrix, "np.random.randint")
# It appears the range of the randint affects the time to solve the equation



# c)

# A * x = y
# A^-1 * A * x = A^-1 * y
# x = A^-1 * y


def inverse_solve(A, y, matrix_method):

    start_time = time.time()
    # Multiply both sides of equation Ax = y
    # by the inverse of A to get x
    x = np.matmul(LA.inv(A), y)
    end_time = time.time()
    print(f"It took {end_time - start_time} seconds using {matrix_method} and the inverse solution!")

    # Same function for verifying the solution is used
    # here as in b)
    verify_solution(A, x, y)

# Test with np.random.rand
A = np.random.rand(100, 100)
y = np.random.rand(100, 1)
inverse_solve(A, y, "np.random.rand")

# Test with np.random.randint
A = np.array([[np.random.randint(0, 1_000_000_000) for _ in range(100)] for _ in range(100)])
y = np.array([np.random.randint(0, 1_000_000_000) for _ in range(100)])
inverse_solve(A, y, "np.random.randint")

# Using LA.inv appears to be a lot faster than 
# row reduced echelon form for solving large matrix equations!
