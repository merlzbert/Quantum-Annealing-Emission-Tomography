from tomography_radon import get_system, get_sinogram, get_reconstruction_circle, get_sinogram_noise, get_rmse, plot_image_sinogram, plot_qa_reconstruction, plot_fbp_reconstruction, plot_sart_reconstruction, get_fbp_reconstruction, get_sart_reconstruction
from tomography_cqm import get_cqm_integer, sample_cqm, get_cqm_reconstructions, save_sampleset
import numpy as np
import dimod
import os
from skimage import io
from skimage.transform import resize
from skimage.filters import threshold_mean
from skimage.data import shepp_logan_phantom
from sklearn import datasets


noise = False
base_path = 'DWave/Results/UnderdeterminedResultsDigits2/'
# Load random incices for samples of the Digit Dataset
# rand_idx = np.random.randint(0, 1797, size=32)
# np.save(base_path + 'digit_idx.npy', rand_idx)
rand_idx = np.load(base_path + 'digit_idx.npy')

results_path = 'DWave/Results/UnderdeterminedResultsDigits/'
# Image size
m = 8
# Load the digits dataset
digits = datasets.load_digits()
no_angles = 2
system = get_system(np.zeros((m, m)), no_angles=no_angles)

print("Computed system matrix")
for i in rand_idx[4:]:
    # Path names ...
    cur_result = results_path + '/' + str(i) + '/'
    isExist = os.path.exists(cur_result)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(cur_result)
    image = digits.images[i]
    mask = get_reconstruction_circle(image.shape).astype(int)
    image = image * mask
    # Forward projection
    if noise:
        sinogram = get_sinogram_noise(image)
    else:
        sinogram = get_sinogram(image, system)
    # Save sinogram
    save_sinogram = cur_result + 'sinogram' + '_' + str(i) + '_' + str(m) + '.png'
    plot_image_sinogram(image, sinogram, save_file = save_sinogram)
    # Set up problem
    bqm = get_cqm_integer(system, sinogram, lowerBound=0, upperBound=17)

    # FBP
    save_fbp = cur_result + 'fbp' + '_' + str(i) + '_' + str(m) + '.png'
    reconstruction_fbp = get_fbp_reconstruction(sinogram, image.shape, no_angles=no_angles)
    plot_fbp_reconstruction(reconstruction_fbp, image, save_file=save_fbp)
    # SART
    save_sart = cur_result + 'sart' + '_' + str(i) + '_' + str(m) + '.png'
    reconstruction_sart = get_sart_reconstruction(sinogram, image.shape, no_angles=no_angles)
    reconstruction_sart = get_sart_reconstruction(sinogram, image.shape, image = reconstruction_sart, no_angles=no_angles)
    plot_sart_reconstruction(reconstruction_sart, image, save_file=save_sart)
    # Sampleset
    title = 'integer_noise' + '_' + 'digit' + '_'  +str(i) + '_' + str(m)
    sampleset = sample_cqm(bqm, label=title)
    save_sampleset(sampleset, cur_result + title + '.json')
    # QA
    reconstruction_qa = get_cqm_reconstructions(sampleset, image.shape)
    save_qa = cur_result + 'qa' + '_' + str(i) + '_' + str(m) + '.png'
    plot_qa_reconstruction(reconstruction_qa, image, save_file=save_qa)

    # Calculate RMSE
    rmse_fbp = get_rmse(reconstruction_fbp, image)
    rmse_sart = get_rmse(reconstruction_sart, image)
    rmse_qa = get_rmse(reconstruction_qa[0].reshape(image.shape), image)
    rmse = np.array([rmse_fbp, rmse_sart, rmse_qa])

    # Save results
    np.savez(cur_result + title + '.npz', image=image, sinogram=sinogram, reconstruction_fbp=reconstruction_fbp,
        reconstruction_sart=reconstruction_sart, reconstruction_qa=reconstruction_qa, rmse=rmse)



