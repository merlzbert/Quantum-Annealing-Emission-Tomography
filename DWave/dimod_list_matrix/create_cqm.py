import dimod

from dimod_list_matrix.data_structures import Matrix, Vector, var_Vector

def compute_norm(M: Matrix, x: var_Vector, y: Vector) -> str:
    matmul = lambda A, B: [" + ".join([str(m) + "*" + x for m, x in zip(row, B)]) for row in A]
    norm = matmul(M.Matrix, x.var_Vector)

    for i in range(len(norm)):
        # drop zero entries
        norm[i] = drop_zeros(norm[i])
        # add the y 
        if y.Vector[i] < 0:
            norm[i] = norm[i] + " + " + str(abs(y.Vector[i]))
        elif y.Vector[i] == 0:
            pass
        else:
            norm[i] = norm[i] + " - " + str(y.Vector[i])
        # square 
        norm[i] = "(" + norm[i]  + ")**2"
    return " + ".join(norm)

def compute_cqm_syntax(result:str, x: Matrix) -> list:
    '''
    Make it compatible with lists
    '''
    non_zero_entries = []
    for i in reversed(range(len(x))):
        tmp_res = result
        result = result.replace('x'+str(i), 'x['+str(i)+']')
        if result != tmp_res:
            non_zero_entries.append(i)
    return result, non_zero_entries

def create_model(problem, non_zeros):
    length = len(non_zeros)
    # Create the quadratic model
    cqm = dimod.ConstrainedQuadraticModel()

    # Create a list of integer variables for each solution component
    x = []
    for i in range(length):
        x.append(dimod.Integer('x'+str(i)))
    # Reformulated objective function to minimize here
    program = 'cqm.set_objective(' + problem + ')'   # das ist absichtlich ein string
    print(program)
    exec(program)
    return cqm

def create_cqm_in_place(M: Matrix, x_vec: var_Vector, y: Vector) -> dimod.ConstrainedQuadraticModel:
    matmul = lambda A, B: [" + ".join([str(t) + "*" + z for t, z in zip(row, B)]) for row in A]

    x_list = x_vec.var_Vector
    for i in range(len(x_list)):
        x_list[i] = x_list[i].replace('x'+str(i), 'x['+str(i)+']')
    norm = matmul(M.Matrix, x_list)

    cqm = dimod.ConstrainedQuadraticModel()
    # Create a list of integer variables for each solution component
    x = []
    for i in range(len(x_list)):
        x.append(dimod.Integer('x'+str(i)))

    drop_entries = []
    for i in range(len(norm)):
        # drop zero entries
        norm[i] = drop_zeros(norm[i])
        if norm[i] == "EMPTY_ROW":
            #TODO: Question: Merlin fragen was hier machen 
            assert y.Vector[i] == 0, "A row in M*x equals to 0, but the corresponding y entry is not equal zero"
            drop_entries.append(i)
            continue
        
        # add the y 
        if y.Vector[i] < 0:
            norm[i] = norm[i] + " + " + str(abs(y.Vector[i]))
        elif y.Vector[i] == 0:
            pass
        else:
            norm[i] = norm[i] + " - " + str(y.Vector[i])
        # square 
        norm[i] = "(" + norm[i]  + ")**2"
    
    for entry in reversed(drop_entries): # iterate backwards to not mess up the norm entry order
        del norm[entry]

    result = " + ".join(norm)
    program = 'cqm.set_objective(' + result + ')'   # das ist absichtlich ein string
    exec(program)

    return cqm


def drop_zeros(row: str) -> str:
    entries = row.split("+")
    result = ""
    for entry in entries:
        entry_cp = entry.strip()
        if entry_cp[0] == str(0):
            # drop the entry
            pass
        elif entry_cp[0] == "-": 
            result += entry
        else:
            result += "+" + entry

    if len(result) == 0:
        return "EMPTY_ROW"

    if result[0] == "+":
        return result[1:]
    else:
        return result
