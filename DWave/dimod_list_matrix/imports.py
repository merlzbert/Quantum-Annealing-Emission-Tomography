# Classical numerical library
import numpy as np

import os 
import errno

import dimod_list_matrix.data_structures as data_structures

def load_matrices_as_list(input_file: str) -> tuple[list, list, list]:
    """
    Load two numpy arrays from a .npz file  

    Create three lists from a .npz file containing exactly two numpy arrays. The .npz file should contain a Matrix M and 
    a vector y, which define a equation Mx = y. 

    Parameters
    ----------
    input_file : str
        filepath to a .npz file containing two numpy arrays
    
    Returns
    -------
    tuple[list, list, list]
        List tuple containing [M, x, y] with M and the Vector y given by the provided numpy array.
        List x is generated in the form x_0...x_n. The dimension n is given by the shape of M.
        
    Raises
    ------
    FileNotFoundError:
        When the given input_file doesn't point to a file
    TypeError:
        When the given input_file doesn't point to a .npz file
    KeyError:
        When the given input_file doesn't contain a numpy array M and a numpy array y
    """
    if not os.path.exists(input_file): 
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_file)

    if not input_file.endswith('.npz'):
        raise TypeError("The given file_path is not a npz file")

    data = np.load(input_file)
    try:
        M = data['M'].tolist()
        y = data['y'].tolist()
    except KeyError:
        print("The numpy arrays in the given file are not named M and y")
        raise KeyError(errno.ENOENT, os.strerror(errno.ENOENT), input_file)
    except Exception as ex:
        raise Exception("An undefined error occured with type " + type(ex) + " in imports.load_matrices_as_list.")
    
    # create the x vector
    x_list = []
    for i in range(data['M'].shape[0]):
        x_list.append((str('x' + str(i))))

    return M, x_list, y 


def load_matrices(input_file: str) -> tuple[data_structures.Matrix, data_structures.var_Vector, data_structures.Vector]:
    """
    Load two numpy arrays from a .npz file  

    Create the three necessary data structures for an equation Mx = y from a .npz file.
    The file must contain exactly two numpy arrays, a Matrix M named M and a vector y named y, 
    which define an equation Mx = y. 

    Parameters
    ----------
    input_file : str
        filepath to a .npz file containing two numpy arrays
    
    Returns
    -------
    tuple[data_structures.Matrix, data_structures.var_Vector, data_structures.Vector]
        The Matrix contains M and the Vector y given by the provided numpy array.
        The var_Vector is the x vector generated in the form x_0...x_n. The dimension m is given by the 
        shape of M.
    
    Raises
    ------
    FileNotFoundError:
        When the given input_file doesn't point to a file
    TypeError:
        When the given input_file doesn't point to a .npz file
    KeyError:
        When the given input_file doesn't contain a numpy array M and a numpy array y 
    """
    if not os.path.exists(input_file): 
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_file)

    if not input_file.endswith('.npz'):
        raise TypeError("The given file_path is not a npz file")
        
    data = np.load(input_file)
    try:
        M = data_structures.Matrix(data['M'].tolist())
        y = data_structures.Vector(data['y'].tolist())
    except KeyError:
        print("The numpy arrays in the given file are not named M and y")
        raise KeyError(errno.ENOENT, os.strerror(errno.ENOENT), input_file)
    except Exception as ex:
        raise Exception("An undefined error occured with type " + type(ex) + " in imports.load_matrices.")

    x_list = []
    for i in range(data['M'].shape[0]):
        x_list.append((str('x' + str(i))))
    x = data_structures.var_Vector(x_list)

    return M, x, y 

def get_x(dim: int) -> data_structures.var_Vector:
    x_list = []
    for i in range(dim):
        x_list.append((str('x' + str(i))))
    
    return data_structures.var_Vector(x_list)
    
def create_data(dim: int, lower_bound: int = -5, upper_bound: int = 5):
    path = "equations/" + str(dim) + "x" + str(dim) + ".npz"
    if not os.path.isdir("equations"):
        os.mkdir("equations")
    if os.path.isfile(path) and path.endswith('.npz'):
        return load_matrices(path)
    else:
        main_generator(dim, path, lower_bound, upper_bound)
        return load_matrices(path)
        

def main_generator(dim: int, path:str, lower_bound: int, upper_bound):
    M, x = create_M_x(dim, lower_bound, upper_bound)
    y = np.matmul(M, x)
    np.savez(path, M=M, y=y)
  
def create_M_x(dim: int, lower_bound, upper_bound) -> tuple[np.ndarray, np.ndarray]:
    M = np.random.randint(lower_bound, upper_bound, size=(dim, dim))
    x = np.random.randint(lower_bound, upper_bound, size=dim)
    return M, x
