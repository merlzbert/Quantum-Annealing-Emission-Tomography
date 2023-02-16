import dimod_list_matrix.imports as imports
import dimod_list_matrix.data_structures as data_structures
import dimod_list_matrix.create_cqm as create_cqm


import dimod
import numpy as np


def main():
    solve_linear_equation()
     
def solve_linear_equation():
    # create and load the Matrices and Vectors for Mx = y with the given dimension
    M, x, y = imports.create_data(dim=5)
    cqm = create_cqm.create_cqm_in_place(M, x, y)
    print(cqm.objective)

def test_null_row():
    M = data_structures.Matrix(np.array([[1, -1, 2], [0, 0, 0], [3, 4, 0]]))
    y = data_structures.Vector(np.array([1, 0, -1]))
    x = imports.get_x(3)

    cqm = create_cqm.create_cqm_in_place(M, x, y)
    print(cqm.objective)

if __name__ == '__main__':
    main()