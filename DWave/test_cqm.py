from tomography_radon import get_system, get_sinogram, plot_image_sinogram, plot_qa_reconstruction
from tomography_cqm import get_cqm_integer, sample_cqm, get_cqm_reconstructions, save_sampleset
import numpy as np
import dimod

image = np.array([[0, 0, 0, 0, 0], [0, 4.0, 0.0, 0, 0], [0, 5.0, 0, 0.0, 0], [0, 6.0, 7.0, 8.0, 0], [0, 0, 0, 0, 0]])

system = get_system(image)
sinogram = get_sinogram(image, system)

plot_image_sinogram(image, sinogram)

cqm = get_cqm_integer(system, sinogram)

bqm, invert = dimod.cqm_to_bqm(cqm)
print(bqm)

# from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
# from dwave.embedding.chain_strength import scaled
# import dwave.inspector
# chain_strength = scaled(bqm)
# sampleset = EmbeddingComposite(DWaveSampler()).sample(bqm, num_reads=1000, label="QUBO Matrix Inversion", chain_strength=chain_strength)
# dwave.inspector.show(sampleset)
# save_sampleset(sampleset, "Sampleset_CQM_to_BQM.json")

# print(invert(sampleset.first))
# sampleset = sample_cqm(cqm, "Test")

# reconstruction =get_cqm_reconstructions(sampleset, image.shape)[0]

# plot_qa_reconstruction(reconstruction, image)



