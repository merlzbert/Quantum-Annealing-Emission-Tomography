# Copyright © Siemens Medical Solutions USA, Inc., 2021. All rights reserved. Unauthorized reproduction prohibited.

import numpy as np
from skimage.transform import radon, iradon, iradon_sart, rescale
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import copy
from skimage.transform import resize
from skimage.transform import radon, iradon, iradon_sart, rescale
from skimage.filters import threshold_mean

def radon_matrix(n1, n2, no_p=None):
    """
    This method generates a system matrix using Skimage's radon transform. As the radon transform 
    is a linear transformation we compute system matrix for the standard basis. More to the point,
    we shift a point source (value = 1) through the image and calculate the radon transform. The
    method is therefore slow as n1 and n2 become very large. Nonetheless, we can generate
    system matrices that mimic the function of the radon function in Skimage.
    Args:
        n1 (int): Image size: image.shape[0]
        n2 (int): Image size: image.shape[0]
        no_p (int, optional): Number of projections. Defaults to None and will then choose max(n1, n2)

    Returns:
        system (np.ndarray, float): Contains the system matrix for the radon transform 
                                    Shape: (no_p*max(n1, n2) ,n1*n2).
    """
    # Set total size and number of projection.
    n = n1 * n2
    if no_p is None:
        npro = max(n1, n2)
    else:
        npro = no_p
    # Initalize angles and system matrix
    theta = np.linspace(0., 180., npro, endpoint=False)
    e1 = np.zeros((n1, n2))
    r1 = radon(e1, theta)
    m= r1.shape[0]
    system = np.zeros((m*len(theta), n))
    # Calculate pixels outside of the reconstruction circle of sklearn. 
    # Sklearn defines a circle in the image which is rotated.
    shape_min = min(n1, n2)
    radius = shape_min // 2
    img_shape = np.array((n1, n2))
    coords = np.array(np.ogrid[:n1, :n2],
                        dtype=object)
    dist = ((coords - img_shape // 2) ** 2).sum(0)
    outside_reconstruction_circle = dist > radius ** 2
    valid_indices = np.where(outside_reconstruction_circle == False)
    # Fill system matrix
    for k in range(len(theta)):
        for i, j in zip(valid_indices[0], valid_indices[1]):
            ei = np.zeros((n1,n2)); 
            ei[i, j] = 1
            ri = radon(ei, [theta[k]])
            system[k*m:(k+1)*m,i*shape_min+j] = ri[:, 0]
    return system

def get_system(image, no_angles=None):
    """
    Return system matrix, just calls radon matrix.

    Args:
        image (np.ndarray, int): Ground truth image of object.
        no_angles (int, optional): Number of projections. Defaults to None and will then choose max(image.shape[0], image.shape[0])

    Returns:
        system (np.ndarray, float): Contains the system matrix for the radon transform 
                                    Shape: (no_p*max(image.shape[0], image.shape[1]), image.shape[0]*image.shape[1]).
    """
     # Set total size and number of projection.
    n = image.shape[0] * image.shape[1]
    if no_angles is None:
        npro = max(image.shape[0], image.shape[1])
    else:
        npro = no_angles
    # Initalize angles and system matrix
    theta = np.linspace(0., 180., npro, endpoint=False)
    e1 = np.zeros((image.shape[0], image.shape[1]))
    r1 = radon(e1, theta)
    m= r1.shape[0]
    system = np.zeros((m*len(theta), n))
    # Calculate pixels outside of the reconstruction circle of sklearn. 
    # Sklearn defines a circle in the image which is rotated.
    shape_min = min(image.shape[0], image.shape[1])
    radius = shape_min // 2
    img_shape = np.array((image.shape[0], image.shape[1]))
    coords = np.array(np.ogrid[:image.shape[0], :image.shape[1]],
                        dtype=object)
    dist = ((coords - img_shape // 2) ** 2).sum(0)
    outside_reconstruction_circle = dist > radius ** 2
    valid_indices = np.where(outside_reconstruction_circle == False)
    # Fill system matrix
    for k in range(len(theta)):
        for i, j in zip(valid_indices[0], valid_indices[1]):
            ei = np.zeros((image.shape[0], image.shape[1])); 
            ei[i, j] = 1
            ri = radon(ei, [theta[k]])
            system[k*m:(k+1)*m,i*shape_min+j] = ri[:, 0]
    return system

def get_sinogram(image, system):
    """
    Return sinogram provided the initial image and system.

    Args:
        image (np.ndarray, int): Ground truth image of object.
        system (np.ndarray, float): Contains the system matrix for the radon transform 
                                    Shape: (no_p*max(image.shape[0], image.shape[1]), image.shape[0]*image.shape[1]).

    Returns:
        sinogram (np.ndarray, float): Flattened sinogram , resulting from the multiplication of the systen matrix with the flattened image. Contains the projections for each angle.
                                      Shape: (no_p*max(image.shape[0], image.shape[1]))
    """
    sinogram = system @ image.flatten()
    return sinogram


def generate_noise(image, lowerBound=-1, upperBound=1):
    """
    Generates ternary noise for a given image. A pixel with a non-zero value is altered by either {-1, 0, 1}.
    Args:
        image (np.ndarray, int): Ground truth image of object.
        lowerBound (int, optional): Lower bound of per pixel noise. Defaults to -1.
        upperBound (int, optional): Upper bound of per pixel noise. Defaults to 1.

    Returns:
        noise_image(np.ndarray, int): Image corrupted with noise.
    """
    noise = np.random.randint(lowerBound, upperBound+1, size=image.shape)
    mask = copy.deepcopy(image)
    mask[mask != 0] = 1
    noise = mask * noise
    noise_image = image + noise
    return noise_image

def get_sinogram_noise(image, no_angles=None):
    """
    Create sinogram by creating noise on the image for each projection.

    Args:
        image (np.ndarray, int): Ground truth image of object.
        no_angles (int, optional): Number of projections. Defaults to None and will then choose max(image.shape[0], image.shape[0])

    Returns:
        sinogram (np.ndarray, float): Flattened sinogram with noise model applied to each projection. Contains the projections for each angle.
                                      Shape: (no_p*max(image.shape[0], image.shape[1]))
    """
    if no_angles is None:
        no_angles = max(image.shape)
    sinogram = np.zeros((max(image.shape), no_angles))
    theta = np.linspace(0., 180., no_angles, endpoint=False)
    for i in range(len(theta)):
        noisy_image = generate_noise(image)
        sinogram[..., i] = radon(noisy_image, theta=[theta[i]]).flatten()
    sinogram = sinogram.flatten('F')
    return sinogram

def get_fbp_reconstruction(sinogram, image_shape, no_angles=None):
    """
    Obtain reconstruction by conventional Filtered Backprojection (FBP) via the iradon transform provided in Skimage.

    Args:
        sinogram (np.ndarray, float): Flattened sinogram , resulting from the multiplication of the systen matrix with the flattened image. Contains the projections for each angle.
                                      Shape: (no_p*max(image.shape[0], image.shape[1]))
        image_shape (tuple, int): Contains the size of the image to reconstruct.
        no_angles (int, optional): Number of projections. Defaults to None and will then choose max(image.shape[0], image.shape[1])

    Returns:
        reconstruction_fbp (np.ndarray, float): Contains the FBP reconstruction from the given sinogram.
    """
    if no_angles is None:
        plt_sinogram = sinogram
        no_angles = max(image_shape)
    else:
        plt_sinogram = sinogram.reshape((image_shape[0], no_angles), order='F')
    reconstruction_fbp = iradon(plt_sinogram, theta=np.linspace(0., 180., no_angles, endpoint=False), filter_name='ramp')
    return reconstruction_fbp

def get_sart_reconstruction(sinogram, image_shape, no_angles=None, image=None):
    if no_angles is None:
        plt_sinogram = sinogram.reshape(image_shape, order='F')
        no_angles = max(image_shape)
    else:
        plt_sinogram = sinogram.reshape((image_shape[0], no_angles), order='F')
    if image is None:
        reconstruction_sart = iradon_sart(plt_sinogram, theta=np.linspace(0., 180., no_angles, endpoint=False))
    else:
        reconstruction_sart = iradon_sart(plt_sinogram, theta=np.linspace(0., 180., no_angles, endpoint=False), image=image)
    return reconstruction_sart

def get_reconstruction_circle(image_shape):
    n1 = image_shape[0]
    n2 = image_shape[1]
    shape_min = min(n1, n2)
    radius = shape_min // 2
    img_shape = np.array((n1, n2))
    coords = np.array(np.ogrid[:n1, :n2], dtype=object)
    dist = ((coords - img_shape // 2) ** 2).sum(0)
    outside_reconstruction_circle = dist > radius ** 2
    outside_reconstruction_circle = np.invert(outside_reconstruction_circle)
    return outside_reconstruction_circle

def get_rmse(reconstruction, image):
    """
    Calculate root mean square error between reconstruction and image.

    Args:
        reconstruction (np.ndarray, int/float): Reconstructed image, value type depends on reconstruction method.
        image (np.ndarray, int):  image (np.ndarray, int): Ground truth image of object.

    Returns:
        rmse (float): Root mean square error.
    """
    error = reconstruction - image
    rmse = np.sqrt(np.mean(error**2))
    return rmse

def get_ssim(reconstruction, image):
    """
    Calculate structural similarity index between reconstruction and image.

    Args:
        reconstruction (np.ndarray, int/float): Reconstructed image, value type depends on reconstruction method.
        image (np.ndarray, int):  image (np.ndarray, int): Ground truth image of object.

    Returns:
        ssim (float): Structural similarity index.
    """
    return structural_similarity(reconstruction, image)

def plot_image_sinogram(image, sinogram, save_file=None):
    """Plot the image and sinogram.

    Args:
        image (np.ndarray, int): Ground truth image of object.
        sinogram (np.ndarray, float): Flattened sinogram , resulting from the multiplication of the systen matrix with the flattened image. Contains the projections for each angle.
                                      Shape: (no_p*max(image.shape[0], image.shape[1]))
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    ax1.set_title("Original")
    ax1.imshow(image, cmap=plt.cm.Greys_r)

    plt_sinogram = sinogram.reshape((image.shape[0], image.shape[0]), order='F')
    dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / plt_sinogram.shape[0]

    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection position (pixels)")

    ax2.imshow(plt_sinogram, cmap=plt.cm.Greys_r,
            extent=(-dx, 180.0 + dx, -dy, plt_sinogram.shape[0] + dy),
            aspect='auto')
    fig.tight_layout()
    fig.patch.set_facecolor('white')
    if save_file is None:
        plt.show()
    else: 
        plt.savefig(save_file)
        plt.close(fig)

def plot_fbp_reconstruction(reconstruction_fbp, image, save_file=None):
    """Plot the reconstruction and error.

    Args:
        image (np.ndarray, int): Ground truth image of object.
        reconstruction_fbp (np.ndarray, float): FBP reconstructed image.
    """
    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), sharex=True, sharey=True)
    ax1.set_title("Reconstruction\nFiltered back projection")
    shw0 = ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    ax2.set_title("Reconstruction error\nFiltered back projection")
    shw1 = ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
    fig.patch.set_facecolor('white')
    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(shw1, cax=cbar_ax, fraction=0.0046)
    if save_file is None:
        plt.show()
    else: 
        plt.savefig(save_file)
        plt.close(fig)

def plot_sart_reconstruction(reconstruction_sart, image, save_file=None):
    """Plot the reconstruction and error.

    Args:
        image (np.ndarray, int): Ground truth image of object.
        reconstruction_fbp (np.ndarray, float): FBP reconstructed image.
    """
    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), sharex=True, sharey=True)
    ax1.set_title("Reconstruction\nSART")
    shw0 = ax1.imshow(reconstruction_sart, cmap=plt.cm.Greys_r)
    ax2.set_title("Reconstruction error\nSART")
    shw1 = ax2.imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r, **imkwargs)
    fig.patch.set_facecolor('white')
    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(shw1, cax=cbar_ax, fraction=0.0046)
    if save_file is None:
        plt.show()
    else: 
        plt.savefig(save_file)
        plt.close(fig)

def plot_qa_reconstruction(reconstruction_qa, image, save_file=None):
    """Plot the reconstruction and error.

    Args:
        image (np.ndarray, int): Ground truth image of object.
        reconstruction_qa (np.ndarray, int): Quantum annealing reconstructed image.
    """
    reconstruction_qa = reconstruction_qa[0].reshape(image.shape)
    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), sharex=True, sharey=True)
    ax1.set_title("Reconstruction\nQuantum Annealing")
    shw0 = ax1.imshow(reconstruction_qa, cmap=plt.cm.Greys_r)
    ax2.set_title("Reconstruction error\nQuantum Annealing")
    shw1 = ax2.imshow(reconstruction_qa - image, cmap=plt.cm.Greys_r, **imkwargs)
    fig.patch.set_facecolor('white')
    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(shw1, cax=cbar_ax, fraction=0.0046)
    if save_file is None:
        plt.show()
    else: 
        plt.savefig(save_file)
        plt.close(fig)

def rebin_data(
    data,
    target_dimension: int
):
  """
  Function that rebins the data object to a smaller size, either by summing or averaging.

  Parameters
  ----------
  data : Data
    Input data object
  target_dimension : Tuple[int, int]
    Target dimension
  method : str, optional
    Method to use, by default 'sum'
  
  Returns
  -------
  Data
    Output data object
  """

  output_data = copy.deepcopy(data)

  for dimension in [-2]:
    if output_data.shape[dimension] > target_dimension:
      dimensions = list(output_data.shape)
      dimensions[dimension] = int(output_data.shape[dimension] / target_dimension)
      dimensions.insert(dimension, int(target_dimension))
      output_data = np.sum(np.reshape(output_data, dimensions), dimension)

  return output_data

def downsample_image(image, new_shape):
  image = resize(image, new_shape, anti_aliasing=True)
  mask = get_reconstruction_circle(image.shape).astype(int)
  thresh = threshold_mean(image)
  image = image > thresh
  image = image * mask
  return image

def upsample_image(image, new_shape):
  image = resize(image, new_shape, anti_aliasing=True, order=0)
  thresh = threshold_mean(image)
  image = image > thresh
  return image

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