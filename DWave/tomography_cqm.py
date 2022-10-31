# Classical numerical library
import numpy as np
# Symbolic matrix computations
from sympy import sympify, Matrix
# Constrained quadratic model
from dimod import ConstrainedQuadraticModel, Integer, Real
# Hybrid quantum annealing sampler
from dwave.system import LeapHybridCQMSampler
# Json
import json
import copy


def get_cqm_integer(system, sinogram, lowerBound = 0, upperBound=50):
    """
    Create constrained quadratic model from linear equation with system and sinogram. This method formulates
    a ConstrainedQuadraticModel suitable for the DWave HybridSampler. Essentially, we minimize a set of linear
    equations specified by the system and sinogram in a least-square fashion. As the HybridSampler is not yet
    able to process matrix operations we pre-formulate the problem in Sympy. We can then set the objective function
    to minimize for the CQM and can further definer constraints. Here we define a non-negativity constraint, a range
    constraint aswell as a zero equality constraint for the pixel outside of the reconstruction circle.
    Args:
        system (np.ndarray, float): Contains the system matrix for the radon transform 
                                    Shape: (no_p*max(image.shape[0], image.shape[1]), image.shape[0]*image.shape[1]).
        sinogram (np.ndarray, float): Flattened sinogram , resulting from the multiplication of the systen matrix with the flattened image. Contains the projections for each angle.
                                      Shape: (no_p*max(image.shape[0], image.shape[1]))
        lowerBound (int, optional): Non-negativity constraint, integer values must be greate than zero. Defaults to 0.
        upperBound (int, optional): Upper bound for integers possible in reconstruction. Defaults to 50.

    Returns:
        cqm (dimod.ConstrainedQuadraticModel): The resulting constrained quadratic model for the HybridSampler.
    """
    from sympy import sympify, Matrix
    # Matrix formulation has to be done in Sympy
    M = Matrix(system.tolist())
    x_list = []
    for i in range(system.shape[1]):
        x_list.append(sympify(str('x'+str(i))))
    X = Matrix(x_list)
    Y = Matrix(sinogram.tolist())
    # Least squares like equation
    A = M.multiply(X)-Y
    result = str(A.T.multiply(A)[0, 0])
    print(result)
    # Make it compatible with lists
    non_zero_entries = []
    for i in reversed(range(len(X))):
        tmp_res = result
        result = result.replace('x'+str(i), 'x['+str(i)+']')
        if result != tmp_res:
            non_zero_entries.append(i)

    # Create the quadratic model
    cqm = ConstrainedQuadraticModel()
    # Create a list of integer variables for each solution component
    x = []
    for i in range(len(X)):
        x.append(Integer('x'+str(i)))
    # Reformulated objective function to minimize here
    program = 'cqm.set_objective(' + result + ')'
    exec(program)
    # Add constraints, here non-negativity and range constraint
    for c in range(len(X)):
        if c in non_zero_entries:
            cqm.add_constraint(x[c] >= lowerBound, label='constraint_' + str(2*c))
            cqm.add_constraint(x[c] <= upperBound, label='constraint_' + str(2*c+1))
        else: 
            cqm.add_constraint(x[c] == 0, label='constraint_' + str(len(X)*2+(c+1)))

    # Substitute integers with self loops
    cqm.substitute_self_loops()
    return cqm

def get_cqm_integer_separatevars(system, sinogram, lowerBound = 0, upperBound=50):
    """
    Create constrained quadratic model from linear equation with system and sinogram. This method formulates
    a ConstrainedQuadraticModel suitable for the DWave HybridSampler. Essentially, we minimize a set of linear
    equations specified by the system and sinogram in a least-square fashion. As the HybridSampler is not yet
    able to process matrix operations we pre-formulate the problem in Sympy. We can then set the objective function
    to minimize for the CQM and can further definer constraints. Here we define a non-negativity constraint, a range
    constraint aswell as a zero equality constraint for the pixel outside of the reconstruction circle.
    Args:
        system (np.ndarray, float): Contains the system matrix for the radon transform 
                                    Shape: (no_p*max(image.shape[0], image.shape[1]), image.shape[0]*image.shape[1]).
        sinogram (np.ndarray, float): Flattened sinogram , resulting from the multiplication of the systen matrix with the flattened image. Contains the projections for each angle.
                                      Shape: (no_p*max(image.shape[0], image.shape[1]))
        lowerBound (int, optional): Non-negativity constraint, integer values must be greate than zero. Defaults to 0.
        upperBound (int, optional): Upper bound for integers possible in reconstruction. Defaults to 50.

    Returns:
        cqm (dimod.ConstrainedQuadraticModel): The resulting constrained quadratic model for the HybridSampler.
    """
    from sympy import sympify, Matrix, separatevars
    # Matrix formulation has to be done in Sympy
    M = Matrix(system.tolist())
    x_list = []
    for i in range(system.shape[1]):
        x_list.append(sympify(str('x'+str(i))))
    X = Matrix(x_list)
    Y = Matrix(sinogram.tolist())
    # Least squares like equation
    A = M.multiply(X)-Y
    result = str(A.T.multiply(A)[0, 0])
    print("Original:\n")
    print(result)
    result = separatevars(result)
    print("Seperate vars:\n")
    print(result)
    # Make it compatible with lists
    non_zero_entries = []
    for i in reversed(range(len(X))):
        tmp_res = result
        result = result.replace('x'+str(i), 'x['+str(i)+']')
        if result != tmp_res:
            non_zero_entries.append(i)

    # Create the quadratic model
    cqm = ConstrainedQuadraticModel()
    # Create a list of integer variables for each solution component
    x = []
    for i in range(len(X)):
        x.append(Integer('x'+str(i)))
    # Reformulated objective function to minimize here
    program = 'cqm.set_objective(' + result + ')'
    exec(program)
    # Add constraints, here non-negativity and range constraint
    for c in range(len(X)):
        if c in non_zero_entries:
            cqm.add_constraint(x[c] >= lowerBound, label='constraint_' + str(2*c))
            cqm.add_constraint(x[c] <= upperBound, label='constraint_' + str(2*c+1))
        else: 
            cqm.add_constraint(x[c] == 0, label='constraint_' + str(len(X)*2+(c+1)))

    # Substitute integers with self loops
    cqm.substitute_self_loops()
    return cqm

def sample_cqm(cqm, time_limit=None, label=None):
    """
    Sample the ConstrainedQuadraticModel on one of DWaves Hybrid Samplers and filter for feasible results.
    Args:
        cqm (dimod.ConstrainedQuadraticModel): Model to store the constrained quadratic model.
        https://docs.ocean.dwavesys.com/en/stable/docs_dimod/reference/constrained.html#dimod.ConstrainedQuadraticModel
        label (string, optional): Label of the CQM problem. Defaults to None.

    Returns:
        sampleset (dimod.SampleSet): Sampleset contatining the returned samples with their associated energy.
        https://docs.ocean.dwavesys.com/en/stable/docs_dimod/reference/sampleset.html
    """
    if label is None:
        label="Tomo Inversion Problem"
    if time_limit is None:
        sampleset = LeapHybridCQMSampler().sample_cqm(cqm, label=label)
    else:
        sampleset = LeapHybridCQMSampler().sample_cqm(cqm, time_limit=time_limit, label=label)

    # Filter feasible solutions
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible) 
    return feasible_sampleset

def get_cqm_reconstructions(sampleset, image_shape):
    """
    Obtain the flattened reconstructions from a dimod.sampleset.
    Args:
        sampleset (dimod.SampleSet): _description_
        image_shape (tuple, int): Size of reconstructed image.
    Returns:
        sampleset (dimod.SampleSet): Sampleset contatining the returned samples with their associated energy.
        https://docs.ocean.dwavesys.com/en/stable/docs_dimod/reference/sampleset.html
    """
    image_size = image_shape[0]*image_shape[1]
    quantum_reconstructions = np.zeros(( len(sampleset), image_size,))
    index = np.arange(0, image_size)
    for s, r in zip(sampleset.data(fields=['sample', 'energy'], sorted_by='energy'), range(len(sampleset))):
        for i in index:
            search = 'x' + str(i)
            quantum_reconstructions[r, i] = s.sample[search]
    return quantum_reconstructions

def save_sampleset(sampleset, filename):
    with open(filename, 'w') as json_file:
        json.dump(sampleset.to_serializable(), json_file)