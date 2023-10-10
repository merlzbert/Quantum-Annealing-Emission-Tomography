# Quantum-Computing-Emission-Tomography

This repository contains Python code to simulate and reconstruct emission tomography images with quantum computing based on our paper: "Exploring Limitations of Hybrid Adiabatic Quantum Computing for Emission Tomography Reconstruction". We have added two examples in the form of a Jupyter notebook for the reconstruction of binary images on the quantum annealer as well as code to reconstruct integer valued images. The tomography_radon.py file contains all functions to simulate the images and reconstruct them using conventional algorithms.

## Getting Started

To run the project:

1. **Create a Virtual Environment and Set Up D-Wave Account:**

   - Create a virtual environment for your project to manage dependencies.

   - Sign up for a D-Wave Ocean account if you haven't already (https://docs.ocean.dwavesys.com/en/stable/overview/install.html). Obtain the required API keys and configure them on your local machine (https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html).

2. **Install Dependencies:**

   Install the necessary Python packages by running the following command within your virtual environment:

   ```bash
   pip install -r requirements.txt

2. **Run the examples in the notebooks:**

   - Use a real quantum annealer to reconstruct your image
   - Compare with conventional algorithms
   
