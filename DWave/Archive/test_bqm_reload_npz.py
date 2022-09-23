from tomography_radon import get_system, get_sinogram, get_rmse, get_reconstruction_circle, plot_image_sinogram, plot_qa_reconstruction, plot_fbp_reconstruction, plot_sart_reconstruction, get_fbp_reconstruction, get_sart_reconstruction
from tomography_bqm import get_bqm, sample_bqm, get_bqm_reconstructions, save_sampleset, get_matmul_string, get_bqm_part
import numpy as np
import dimod
import os
from skimage import io
from skimage.transform import resize
from skimage.filters import threshold_mean

mat_sizes = [4, 8, 12, 16, 20, 24, 28, 32]
base_path = 'DWave/BinaryImages'
results_path = 'DWave/Results'
filenames = os.listdir(base_path)
print(filenames)

for m in mat_sizes:
    system = get_system(np.zeros((m, m)))
    print("Computed system matrix")
    for f in filenames:
        # Path names ...
        filename = base_path + '/' + f
        img_name = f[:-4]
        cur_result = results_path + '/' + img_name + '/' + str(m) + '/'
        res_npz_file = [_ for _ in os.listdir(cur_result) if _.endswith(".npz")]
        res_npz = np.load(cur_result + res_npz_file[0])
        # Resize and binarize
        image = res_npz['image']
        mask = get_reconstruction_circle(image.shape)
        print(mask)
        image *= mask
        # Forward projection
        sinogram = res_npz['sinogram']
        # Save sinogram
        save_sinogram = cur_result + 'sinogram' + '_' + img_name + '_' + str(m) + '.png'
        plot_image_sinogram(image, sinogram, save_file = save_sinogram)

        save_fbp = cur_result + 'fbp' + '_' + img_name + '_' + str(m) + '.png'
        reconstruction_fbp = res_npz['reconstruction_fbp'] * mask
        plot_fbp_reconstruction(reconstruction_fbp, image, save_file=save_fbp)
        # SART
        save_sart = cur_result + 'sart' + '_' + img_name + '_' + str(m) + '.png'
        reconstruction_sart = res_npz['reconstruction_sart'] * mask
        plot_sart_reconstruction(reconstruction_sart, image, save_file=save_sart)
        # Sampleset
        title = 'binary' + '_' + f[:-4] + '_' + str(m)
        # QA
        reconstruction_qa = res_npz['reconstruction_qa']
        print(reconstruction_qa.shape)
        save_qa = cur_result + 'qa' + '_' + img_name + '_' + str(m) + '.png'
        plot_qa_reconstruction(reconstruction_qa, image, save_file=save_qa)

        rmse_fbp = get_rmse(reconstruction_fbp, image)
        rmse_sart = get_rmse(reconstruction_sart, image)
        rmse_qa = get_rmse(reconstruction_qa[0].reshape(image.shape)*mask, image)
        rmse = np.array([rmse_fbp, rmse_sart, rmse_qa])

        # np.savez(cur_result + title + '.npz', image=image, sinogram=sinogram, reconstruction_fbp=reconstruction_fbp,
        #  reconstruction_sart=reconstruction_sart, reconstruction_qa=reconstruction_qa, rmse=rmse)


