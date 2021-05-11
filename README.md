# VerifyGAN
Companion code for "Verification of Image-based Controllers Using Generative Models"

S.M. Katz*, A.L. Corso*, C.A. Strong*, M.J. Kochenderfer

## data
The data folder contains the datasets from images of a runway taken from a camera angle of the right wing of a Cessna 208B Grand Caravan taxiing down runway 04 of Grant County International Airport in the [X-Plane 11 flight simulator](https://www.x-plane.com/). The images are downsampled to 8 x 16 images and converted to grayscale.

`SK_DownsampledGANFocusAreaData.h5` - data from a clean 200-meter stretch of the runway generated specifically for the GAN; created by uniformly sampling random locations in the 200-meter stretch. X values are the labels of crosstrack error (meters), heading error (degrees), and downtrack position (meters) and the y values are the images.

To load in Julia:
```julia
using HDF5
X_train = h5read("SK_DownsampledGANFocusAreaData.h5", "X_train") # 3 x 10000
y_train = h5read("SK_DownsampledGANFocusAreaData.h5", "y_train") # 16 x 8 x 10000
```

## models
The control network that takes in a downsampled runway image and predicts crosstrack error and heading error is saved in both the [NNet](https://github.com/sisl/NNet) (`TinyTaxiNet.nnet`) and [Flux](https://fluxml.ai/) (`TinyTaxiNet.bson`) format.

The full models (concatenated generator and control network) for both the supervised and adversarial MLPs are located here as well in both formats. These networks go from two latent variables and the normalized crosstrack and heading error to a prediction of the crosstrack and heading error.

## xplane interface
The python files to interface with the XPlane simulator are contained in the `src/xplane_interface` folder. More information about the `xpc3.py` file created by NASA X-Plane Connect can be found [here](https://github.com/nasa/XPlaneConnect). Other notable files include `genGANData.py` and `SK_genTrainingData.py`, which are used to generate and downsample the GAN training data respectively. The `sim_network.py` file allows us to simulate our simple dynamics model using X-Plane 11 images to drive the controller.

## gan training
The code for training the GANs as well as some of the saved generators can be found in `src/gan_training`. The file `cGAN_common.jl` contains functions for training a conditional GAN, the file `spectral_norm.jl` implements spectral normalization layers in Flux to be used by the discriminator, and the file `taxi_models_and_data` implements the generator and discriminator architectures used for the taxinet problem. `train_gans.jl` runs the code for training a GAN with various hyperparameters. The `train_smaller_generator` file contains the code to train a smaller generator in a supervised learning fashion.

To run the training code, ensure that all necessary julia packages are installed and then run:
```julia
include("train_gans.jl")
```
This code was developed and tested using Julia 1.5.

The settings data structures allows for easy specification on training settings:

```julia
@with_kw struct Settings
	G # Generator to train
	D # Discriminator to train
	loss # Loss function
	img_fun # Function to load in the image data
	rand_input # Function to generate a random input for the generator
	batch_size::Int = 128 # Batch size of the data
	latent_dim::Int = 100 # Number of latent variables
	nclasses::Int = 2 # Number of input variables (crosstrack error and heading error)
	epochs::Int = 120 # Numer of epochs through the data for training
	verbose_freq::Int = 2000 # How often to print and save training info
	output_x::Int = 6 # Size of image output examples
	output_y::Int = 6 # Size of image output examples
	optD = ADAM(0.0002, (0.5, 0.99))
	optG = ADAM(0.0002, (0.5, 0.99))
	output_dir = "output" # Folder to save outputs to
	n_disc = 1 # Number of discriminator training steps between generator training step
end
```

## verification
The verification code relies on a modified version of Ai2z, which is implemented in the `src/verification/Ai2zPQ.jl` file. The `verify.jl` file located in `src/verification` loads in the necessary files and contains functions for computing the minimum and maximum control output over a given region in the generator's input space and for doing so for each cell in an input space.

To divide up the input space and run verification, run the following lines of code:

```julia
include("verify.jl")

network = read_nnet("../../models/full_mlp_supervised.nnet")

max_widths = [0.2, 0.5] # Maximum cell widths
lbs = [-11.0, -30.0]
ubs = [11.0, 30.0]

tree = create_tree(max_widths, lbs, ubs)
verify_tree!(tree, network)
```

The post-verification tree stuctures (which express the cell discretization as a kdtree) are saved in the `src/verification/verified_trees` folder.

The model checking code for reachability analysis is located in the `src/model_check.jl` file. To run model checking on the tree, run:

```julia
include("model_check.jl")
label_tree_failures!(tree)
model_check!(tree)
```

To run forward reachability analysis, run:

```julia
lbs = [-10.0, -10.0] # Lower bounds of region of start states
ubs = [10.0, 10.0] # Upper bounds of region of start states
label_start_states!(tree, lbs, ubs)
trees = forward_reach(tree)
```

The `viz/` folder contains code for plotting the results.

## gan evaluation
The file `radius.jl` contains code for calculating the Euclidean distance for each training point to the closest generated image. To calculate these distances for a generator network, make sure the correct network is specified on line 50 and run:

```julia
include("radius.jl")
```

The file `approx_radius.jl` contains code for approximating the Euclidean distance for each training point to the closest generated image via sampling. To calculate these distances for a generator network, make sure the correct network is specified on line 50 and run:

```julia
include("approx_radius.jl")
```