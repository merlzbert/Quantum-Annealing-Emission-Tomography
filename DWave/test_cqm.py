from tomography_radon import get_system, get_sinogram, get_reconstruction_circle, get_rmse, plot_image_sinogram, plot_qa_reconstruction, plot_fbp_reconstruction, plot_sart_reconstruction, get_fbp_reconstruction, get_sart_reconstruction
# from tomography_cqm import get_cqm_integer, sample_cqm, get_cqm_reconstructions, save_sampleset
from tomography_cqm_noconstraint import get_cqm_integer, sample_cqm, get_cqm_reconstructions, save_sampleset
import numpy as np
import dimod
import os
from skimage import io
from skimage.transform import resize
from skimage.filters import threshold_mean
from skimage.data import shepp_logan_phantom

# Matrix sizes to test
mat_sizes = [32]
int_size = 16
# Images
base_path = 'DWave/Results/ResultsSheppLoganTimeLimit120'
results_path = 'DWave/Results/ResultsSheppLoganTimeLimit120'
filenames = os.listdir(base_path)
print(filenames)

for m in mat_sizes:
    # Calculate system matrix for each matrix size
    no_angles = m
    system = get_system(np.zeros((m, m)), no_angles=no_angles)
    print("Computed system matrix")
    for f in filenames:
        # Path names ...
        filename = base_path + '/' + f
        img_name = f
        cur_result = results_path + '/' + img_name + '/' + str(m) + '/'
        isExist = os.path.exists(cur_result)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(cur_result)
        # Load shepp Logan phantom
        image = shepp_logan_phantom()
        # Resize and binarize
        image = resize(image, (m, m), anti_aliasing=True)
        image *= int_size
        image = image.astype(int)
        mask = get_reconstruction_circle(image.shape).astype(int)
        image = image * mask
        # Forward projection
        sinogram = get_sinogram(image, system)
        # Save sinogram
        save_sinogram = cur_result + 'sinogram' + '_' + img_name + '_' + str(m) + '.png'
        plot_image_sinogram(image, sinogram, save_file = save_sinogram)
        # Set up problem
        bqm, non_zero_entries = get_cqm_integer(system, sinogram, lowerBound=0, upperBound=int_size)
        # FBP
        save_fbp = cur_result + 'fbp' + '_' + img_name + '_' + str(m) + '.png'
        reconstruction_fbp = get_fbp_reconstruction(sinogram, image.shape, no_angles=no_angles)
        plot_fbp_reconstruction(reconstruction_fbp, image, save_file=save_fbp)
        # SART
        save_sart = cur_result + 'sart' + '_' + img_name + '_' + str(m) + '.png'
        reconstruction_sart = get_sart_reconstruction(sinogram, image.shape, no_angles=no_angles)
        reconstruction_sart = get_sart_reconstruction(sinogram, image.shape, image = reconstruction_sart, no_angles=no_angles)
        plot_sart_reconstruction(reconstruction_sart, image, save_file=save_sart)
        # Sampleset
        title = 'integer' + '_' + f + '_' + str(m)
        sampleset = sample_cqm(bqm, time_limit=30, label=title)
        save_sampleset(sampleset, cur_result + title + '.json')
        # QA
        reconstruction_qa = get_cqm_reconstructions(sampleset, image.shape, non_zero_entries)
        save_qa = cur_result + 'qa' + '_' + img_name + '_' + str(m) + '.png'
        plot_qa_reconstruction(reconstruction_qa, image, save_file=save_qa)

        # Calculate RMSE for each reconstruction technique
        rmse_fbp = get_rmse(reconstruction_fbp, image)
        rmse_sart = get_rmse(reconstruction_sart, image)
        rmse_qa = get_rmse(reconstruction_qa[0].reshape(image.shape), image)
        rmse = np.array([rmse_fbp, rmse_sart, rmse_qa])

        np.savez(cur_result + title + '.npz', image=image, sinogram=sinogram, reconstruction_fbp=reconstruction_fbp,
         reconstruction_sart=reconstruction_sart, reconstruction_qa=reconstruction_qa, rmse=rmse)




