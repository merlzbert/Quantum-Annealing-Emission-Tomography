from tomography_radon import get_system, get_sinogram, get_reconstruction_circle, get_rmse, plot_image_sinogram, plot_qa_reconstruction, plot_fbp_reconstruction, plot_sart_reconstruction, get_fbp_reconstruction, get_sart_reconstruction
from tomography_bqm import get_bqm, sample_bqm, get_bqm_reconstructions, save_sampleset, get_matmul_string, get_bqm_part
import numpy as np
import dimod
import os
from skimage import io
from skimage.transform import resize
from skimage.filters import threshold_mean

mat_sizes = [16, 20, 24, 28, 32]
base_path = 'DWave/BinaryImages'
results_path = 'DWave/Results/UnderdeterminedBinary'
filenames = os.listdir(base_path)

for m in mat_sizes:
    no_angles = m//2
    system = get_system(np.zeros((m, m)), no_angles=no_angles)
    print("Computed system matrix")
    for f in filenames:
        # Path names ...
        filename = base_path + '/' + f
        img_name = f[:-4]
        cur_result = results_path + '/' + img_name + '/' + str(m) + '/'
        image = io.imread(filename)
        # Resize and binarize
        image = resize(image, (m, m), anti_aliasing=True)
        thresh = threshold_mean(image)
        image = image > thresh
        mask = get_reconstruction_circle(image.shape).astype(int)
        image = image * mask
        # Forward projection
        sinogram = get_sinogram(image, system)
        print("Image shape: ", image.shape)
        print("System shape: ", system.shape)
        print("Sinogram shape: ", sinogram.shape)
        # Save sinogram
        save_sinogram = cur_result + 'sinogram' + '_' + img_name + '_' + str(m) + '.png'
        plot_image_sinogram(image, sinogram, save_file = save_sinogram)
        # Set up problem
        bqm = get_bqm(system, sinogram)
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
        title = 'underdetermined_binary' + '_' + f[:-4] + '_' + str(m)
        sampleset = sample_bqm(bqm, label=title)
        save_sampleset(sampleset, cur_result + title + '.json')
        # QA
        reconstruction_qa = get_bqm_reconstructions(sampleset, image.shape)
        save_qa = cur_result + 'qa' + '_' + img_name + '_' + str(m) + '.png'
        plot_qa_reconstruction(reconstruction_qa, image, save_file=save_qa)

        rmse_fbp = get_rmse(reconstruction_fbp, image)
        rmse_sart = get_rmse(reconstruction_sart, image)
        rmse_qa = get_rmse(reconstruction_qa[0].reshape(image.shape), image)
        rmse = np.array([rmse_fbp, rmse_sart, rmse_qa])

        np.savez(cur_result + title + '.npz', image=image, sinogram=sinogram, reconstruction_fbp=reconstruction_fbp,
         reconstruction_sart=reconstruction_sart, reconstruction_qa=reconstruction_qa, rmse=rmse)


