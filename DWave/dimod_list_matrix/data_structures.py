import numpy as np
import errno
import os 

#TODO _setitem__

class var_Vector():
    def __init__(self, n):
        if isinstance(n, int):
            self.var_Vector = self.__create(n)
        elif isinstance(n, list):
            self.var_Vector = n
        else:
            raise ValueError("Please provide a supported data structure, i.e. int, list")


    def __create(self, n:int):
        vec = []
        for i in range(n):
            vec.append("x" + str(i))
        return vec
        
    def __str__(self):
        return str(self.var_Vector)

    def __getitem__(self, key):
        return self.var_Vector[key]

class Vector():
    def __init__(self, v):
        if isinstance(v, np.ndarray):
            if v.ndim != 1: raise ValueError("The provided numpy array has to be one-dimensional")
            self.Vector = v.tolist()
        elif isinstance(v, list):
            self.Vector = v
        else:
            raise ValueError("Please provide a numpy array or list as Matrix input")

    def __str__(self):
        return str(self.Vector)

    def __getitem__(self, key):
        return self.Vector[key]

class Matrix(list):
    """
    Matrix class for fast computations storing the matrices as lists.
    """
    def __init__(self, M):
        if isinstance(M, np.ndarray):
            self.Matrix = M.tolist()
        elif isinstance(M, list):
            self.Matrix = M
        else:
            raise ValueError("Please provide a supported data structure, i.e. numpy.array, list")

    def __str__(self):
        return str(self.Matrix)

    def __len__(self):
        return len(self.Matrix)

    def __getitem__(self, key):
        return self.Matrix[key]

    def __add__(self, B):
        if not isinstance(B, Matrix):
            raise ValueError("Addition of a Matrix is only defined with other matrices")
        
        return Matrix(np.add(np.array(self.Matrix), np.array(B.Matrix)))
    
    def __sub__(self, B):
        if not isinstance(B, Matrix):
            raise ValueError("Subtraction of a Matrix is only defined with other matrices")

        return Matrix(np.subtract(np.array(self.Matrix), np.array(B.Matrix)))

    def __mul__(self, B) -> list:
        """
        Multiply the matrix, which calls its own mult function with B depending on the data type of B.

        Parameters
        ----------
        B : {Matrix, var_Vector, Vector}
            the right side of the multiplication
        
        Returns
        -------
        list
            list containing the result

        Raises
        -------
        ValueError
            if B is not a Matrix, var_Vector or Vector
        """
        if isinstance(B, Matrix):
            return self.__mult_Matrix(B)
        elif isinstance(B, var_Vector):
            return self.__mult_var_Vector(B)
        elif isinstance(B, Vector):
            return self.__mult_Vector(B)
        else:
            raise ValueError("Please provide a supported data structure")

    def __mult_Matrix(self, B):
        return Matrix(np.matmul(np.array(self.Matrix), np.array(B.Matrix)).tolist())

    def __mult_Vector(self, B):
        matmul = lambda A, B: [sum(m*x for m, x in zip(row, B)) for row in A]
        return Vector(matmul(self.Matrix, B))

    def __mult_var_Vector(self, B):
        matmul = lambda A, B: [" + ".join([str(m) + "*" + x for m, x in zip(row, B)]) for row in A]
        
        norm = matmul(self.Matrix, B)
        for i in range(len(norm)):
            norm[i] = self.__drop_zeros(norm[i]) # drop zero entries
        return " + ".join(norm)

    def __drop_zeros(self, row: str) -> str:
        """
        Recieves one row of the multiplication of a Matrix and a var_Vector and drops zero entries and computes signs

        Parameters
        ----------
        row : str
            one Matrix row after the multiplication M*x
        
        Returns
        -------
        str
            the adjusted row 

        Raises
        -------
        ValueError
            if row is not a str
        """
        if not isinstance(row, str):
            raise ValueError("The row has to be a string, but was a " + str(type(row)) + " with the value: " + str(row) + ".")            

        entries = row.split("+")
        result = ""
        for entry in entries:
            entry_cp = entry.strip()
            if entry_cp[0] == str(0):
                pass # drop the entry
            elif entry_cp[0] == "-": 
                result += entry
            else:
                result += "+" + entry

        if len(result) == 0:
            # TODO add error message
            print("NULL")
            exit(0)

        # drop the first + in the string, e.g '3x + 5' instead of '+3x + 5'
        if result[0] == "+":
            return result[1:]
        else:
            return result

        def getElement(self, i, j):
            return self.Matrix[i][j]
   
        def transpose(self):
            return [list(i) for i in zip(*self.Matrix)]

        def transpose_in_place(self):
            self.Matrix = [list(i) for i in zip(*self.Matrix)]
