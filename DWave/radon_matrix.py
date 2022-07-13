import numpy as np
from skimage.transform import radon, rescale

def radon_matrix(n1, n2, no_p=None):
    n = n1 * n2
    if no_p is None:
        npro = max(n1, n2)
    else:
        npro = no_p

    theta = np.linspace(0., 180., npro, endpoint=False)
    e1 = np.zeros((n1, n2))
    r1 = radon(e1, theta)
    m= r1.shape[0]
    R = np.zeros((m*len(theta), n))

    shape_min = min(n1, n2)
    radius = shape_min // 2
    img_shape = np.array((n1, n2))
    coords = np.array(np.ogrid[:n1, :n2],
                        dtype=object)
    dist = ((coords - img_shape // 2) ** 2).sum(0)
    outside_reconstruction_circle = dist > radius ** 2
    valid_indices = np.where(outside_reconstruction_circle == False)

    for k in range(len(theta)):
        for i, j in zip(valid_indices[0], valid_indices[1]):
            ei = np.zeros((n1,n2)); 
            ei[i, j] = 1
            ri = radon(ei, [theta[k]])
            R[k*m:(k+1)*m,i*shape_min+j] = ri[:, 0]
    return R


################################################################
# Test:
    # x = np.random.randint(0, 2, (n1, n2))
    # x[outside_reconstruction_circle] = 0


    # print("X: \n", x)

    # yr = R @ x.flatten()

    # yR = radon(x, theta, preserve_range=True)
    # print("Radon: \n", yR)

    # yr = yr.reshape(yR.shape)
    # print("Mat mul: \n", yR)