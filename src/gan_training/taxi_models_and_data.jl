# Functions for training on taxinet data

using HDF5
include("cGAN_common.jl")

"""
Discriminator Models
"""

## Taxi discriminator with convolutions
struct TaxiDConv
	d_labels		# Submodel to take labels as input and convert them to the shape of image ie. (28, 28, 1, batch_size)
	d_common
end

Flux.trainable(d::TaxiDConv) = (d.d_labels, d.d_common)

function TaxiDConv(s::Settings)
	d_labels = Chain(Dense(s.nclasses,16*8), x-> reshape(x, 8, 16, 1, size(x, 2))) #|> gpu
	d_common = Chain(
					 Conv((3,3), 2=>64, pad=1, stride=1),
					 BatchNorm(64, leakyrelu),
			 		 Conv((4,4), 64=>64, pad=1, stride=2),
					 BatchNorm(64, leakyrelu),
					 Conv((3,3), 64=>128, pad=1, stride=1),
					 BatchNorm(128, leakyrelu),
					 Conv((4,4), 128=>128, pad=1, stride=2),
					 BatchNorm(128, leakyrelu),
					 Conv((3,3), 128=>256, pad=1, stride=1),
					 BatchNorm(256, leakyrelu),
					 Conv((4,4), 256=>256, pad=1, stride=2),
					 BatchNorm(256, leakyrelu),
			  		 Conv((3,3), 256=>512, pad=1, stride=1),
			  		 x -> reshape(x, 1024, :),
			  		 Dense(1024, 1)
					 ) #|> gpu
   TaxiDConv(d_labels, d_common)
end

function (m::TaxiDConv)(x, y)
	t = cat(m.d_labels(y), x, dims=3)
	return m.d_common(t)
end


## spectral
struct TaxiDConvSpectral
	d_labels		# Submodel to take labels as input and convert them to the shape of image ie. (28, 28, 1, batch_size)
	d_common
end

Flux.trainable(d::TaxiDConvSpectral) = (d.d_labels, d.d_common)

function TaxiDConvSpectral(s::Settings)
	d_labels = Chain(DenseSN(s.nclasses,16*8), x-> reshape(x, 8, 16, 1, size(x, 2))) #|> gpu
	d_common = Chain(
					 ConvSN((3,3), 2=>64, pad=1, stride=1, leakyrelu),
			 		 ConvSN((4,4), 64=>64, pad=1, stride=2, leakyrelu),
					 ConvSN((3,3), 64=>128, pad=1, stride=1, leakyrelu),
					 ConvSN((4,4), 128=>128, pad=1, stride=2, leakyrelu),
					 ConvSN((3,3), 128=>256, pad=1, stride=1, leakyrelu),
					 ConvSN((4,4), 256=>256, pad=1, stride=2, leakyrelu),
			  		 ConvSN((3,3), 256=>512, pad=1, stride=1, leakyrelu),
			  		 x -> reshape(x, 1024, :),
			  		 Dense(1024, 1)
					 ) #|> gpu
   TaxiDConvSpectral(d_labels, d_common)
end

function (m::TaxiDConvSpectral)(x, y)
	t = cat(m.d_labels(y), x, dims=3)
	return m.d_common(t)
end

## Taxi discriminator with dense layers
struct TaxiDMLP
	net
end

Flux.trainable(d::TaxiDMLP) = (d.net,)

function TaxiDMLP(s::Settings)
	TaxiDMLP(Chain(Dense(16*8 + s.nclasses, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 1, )) |> gpu)
end

function (m::TaxiDMLP)(x, y)
	m.net(vcat(reshape(x, :, size(y,2)), y))
end

"""
Generator Models
"""

## Taxi generator with conv layers
struct TaxiGConv
    g_labels          # Submodel to take labels as input and convert it to the shape of (7, 7, 1, batch_size) 
    g_latent          # Submodel to take latent_dims as input and convert it to shape of (7, 7, 128, batch_size)
    g_common    
end

Flux.trainable(d::TaxiGConv) = (d.g_labels, d.g_latent, d.g_common)

function TaxiGConv(s::Settings)
	g_labels = Chain(Dense(s.nclasses, 49), x-> reshape(x, 7 , 7 , 1 , size(x, 2))) |> gpu
    g_latent = Chain(Dense(s.latent_dim, 6272), x-> leakyrelu.(x, 0.2f0), x-> reshape(x, 7, 7, 128, size(x, 2))) |> gpu
    g_common = Chain(ConvTranspose((4, 4), 129=>128; stride=2, pad=1),
            BatchNorm(128, leakyrelu),
            Dropout(0.25),
            ConvTranspose((4, 4), 128=>64; stride=2, pad=1),
            BatchNorm(64, leakyrelu),
            Conv((5, 5), 64=>1, tanh; stride=(4, 2), pad=4)) |> gpu
	TaxiGConv(g_labels, g_latent, g_common)
end

function (m::TaxiGConv)(x, y)
    t = cat(m.g_labels(y), m.g_latent(x), dims=3)
    return m.g_common(t)
end

## Taxi generator with mlp
struct TaxiGMLP
	net
end

Flux.trainable(d::TaxiGMLP) = (d.net,)

function BigMLP(s::Settings)
 	TaxiGMLP(Chain(Dense(s.latent_dim + s.nclasses, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 16*8), x -> reshape(x, 8,16,1,:))) #|> gpu)
end

function SmallMLP(s::Settings)
 	TaxiGMLP(Chain(Dense(s.latent_dim + s.nclasses, 128, relu), Dense(128, 128, relu), Dense(128, 16*8), x -> reshape(x, 8,16,1,:)) |> gpu)
end
 
function (m::TaxiGMLP)(x, y)
	m.net(vcat(reshape(x, :, size(y,2)), y))
end

"""
Generating Outputs
"""

# Function that returns random input for the generator
function Taxi_input(s::Settings)
   # x = randn(s.latent_dim, s.batch_size) |> gpu
   x = Float32.(rand(Uniform(-1.0, 1.0), s.latent_dim, s.batch_size)) #|> gpu
   # y = Float32.(vcat(rand(Uniform(-2.0, 2.0), 1, s.batch_size), rand(Uniform(-2.0, 2.0), 1, s.batch_size))) |> gpu
   y = zeros(Float32, 2, s.batch_size) #|> gpu
   x, y
end

function gen_Taxi_images(s::Settings)
	# Load the data
	# fn = "KJ_DownsampledTrainingData.h5"
	# images = h5read(fn, "X_train")
	# y = Float32.(h5read(fn, "y_train"))
	
	fn = "../../data/SK_DownsampledGANFocusAreaData.h5"
	images = h5read(fn, "y_train") # yes I know the labels seem backwards
	println("starting size: ", size(images))
	images = permutedims(images, [2,1, 3])
	println("permuted dims: ", size(images))
	y = h5read(fn, "X_train")[1:2, :]
		println("std1: ", std(y[1,:]), " std2: ", std(y[2,:]))
	y[1,:] ./= std(y[1,:])
	y[2,:] ./= std(y[2,:])
	

	println("extrema of dim1: ", extrema(y[1,:]), " extrema of dim2: ", extrema(y[2,:]))
	#y = y |> gpu

	# Generate fixed set of labels and save the corresponding images
	N = s.output_x * s.output_y
	#fixed_noise = [randn(s.latent_dim, 1) |> gpu for _=1:N]
	fixed_noise = [randn(s.latent_dim, 1) for _=1:N]
	
	indices = rand(MersenneTwister(0), 1:size(y,2), N)
	fixed_labels = [y[:, i] for i in indices]
	real_images = [cpu(images[:,:, i]) for i in indices]
	image_array = permutedims(reduce(vcat, reduce.(hcat, partition(real_images, s.output_y))), (2,1))
	image_array = Gray.(image_array)
	save("output/real_images_sk.png", image_array)

	# rescale the data
	images = reshape(2f0 .* images .- 1f0, 8, 16, 1, :) #|> gpu # Normalize to [-1, 1]
	data = DataLoader((Float32.(images), Float32.(y)), batchsize=s.batch_size, shuffle = true, partial = false)
								 
	data, fixed_noise, fixed_labels
end

epochs = 500
G = train(Settings(G = BigMLP, D = TaxiDConvSpectral, epochs = epochs, rand_input = Taxi_input, loss = DCGANLoss(), img_fun = gen_Taxi_images, nclasses = 2, latent_dim = 2, output_dir = "output/skTaxi_BigMLP_ld2_Spectral_nopretrain"))

"""
Old training runs
"""

# First reduce the latent dimension size
# epochs = 500
# G_dcgan100, D_dcgan100 = train(Settings(G = TaxiGConv, D = TaxiDConvSpectral, epochs = epochs, rand_input = Taxi_input, loss = DCGANLoss(), img_fun = gen_Taxi_images, nclasses = 2, latent_dim = 100, output_dir = "skTaxi_DCGAN_ld100_Spectral"))

# G = TaxiGConv(G_dcgan100.g_labels |> cpu, G_dcgan100.g_latent |> cpu, G_dcgan100.g_common |>cpu)
# @save "dcgan_ld100_generator_permuted.bson" G


# G_dcgan20, D_dcgan20 = train(Settings(G = TaxiGConv, D = TaxiDConvSpectral, epochs = epochs, rand_input = Taxi_input, loss = DCGANLoss(), img_fun = gen_Taxi_images, nclasses = 2, latent_dim = 20, output_dir = "skTaxi_DCGAN_ld20_Spectral"))

# G = TaxiGConv(G_dcgan20.g_labels |> cpu, G_dcgan20.g_latent |> cpu, G_dcgan20.g_common |>cpu)
# @save "dcgan_ld20_generator_permuted.bson" G

# G_dcgan5, D_dcgan5 = train(Settings(G = TaxiGConv, D = TaxiDConvSpectral, epochs = epochs, rand_input = Taxi_input, loss = DCGANLoss(), img_fun = gen_Taxi_images, nclasses = 2, latent_dim = 5, output_dir = "skTaxi_DCGAN_ld5_Spectral"))

# G = TaxiGConv(G_dcgan5.g_labels |> cpu, G_dcgan5.g_latent |> cpu, G_dcgan5.g_common |>cpu)
# @save "dcgan_ld5_generator_permuted.bson" G

# G_dcgan2, D_dcgan2 = train(Settings(G = TaxiGConv, D = TaxiDConvSpectral, epochs = epochs, rand_input = Taxi_input, loss = DCGANLoss(), img_fun = gen_Taxi_images, nclasses = 2, latent_dim = 2, output_dir = "permuted_sk2Taxi_DCGAN_ld2_Spectral"))

# G = TaxiGConv(G_dcgan2.g_labels |> cpu, G_dcgan2.g_latent |> cpu, G_dcgan2.g_common |>cpu)
# @save "dcgan_ld2_generator_permuted.bson" G
# 
# G_dcgan2 = G_dcgan2.net |> cpu
# 
# G = TaxiGConv(G_dcgan2.g_labels |> cpu, G_dcgan2.g_latent |> cpu, G_dcgan2.g_common |>cpu)
# @save "dcgan_generator_permuted.bson" G

# Train MLPs with existing discriminators
# train(Settings(G = BigMLP, D = (s)-> deepcopy(D_dcgan100), epochs = epochs, rand_input = Taxi_input, loss = DCGANLoss(), img_fun = gen_Taxi_images, nclasses = 2, latent_dim = 100, output_dir = "skTaxi_BigMLP_ld100_Spectral"))
# 
# train(Settings(G = SmallMLP, D = (s)-> deepcopy(D_dcgan100), epochs = epochs, rand_input = Taxi_input, loss = DCGANLoss(), img_fun = gen_Taxi_images, nclasses = 2, latent_dim = 100, output_dir = "skTaxi_SmallMLP_ld100_Spectral"))
# 
# train(Settings(G = BigMLP, D = (s)-> deepcopy(D_dcgan20), epochs = epochs, rand_input = Taxi_input, loss = DCGANLoss(), img_fun = gen_Taxi_images, nclasses = 2, latent_dim = 20, output_dir = "skTaxi_BigMLP_ld20_Spectral"))
# 
# train(Settings(G = SmallMLP, D = (s)-> deepcopy(D_dcgan20), epochs = epochs, rand_input = Taxi_input, loss = DCGANLoss(), img_fun = gen_Taxi_images, nclasses = 2, latent_dim = 20, output_dir = "skTaxi_SmallMLP_ld20_Spectral"))
# 
# train(Settings(G = BigMLP, D = (s)-> deepcopy(D_dcgan5), epochs = epochs, rand_input = Taxi_input, loss = DCGANLoss(), img_fun = gen_Taxi_images, nclasses = 2, latent_dim = 5, output_dir = "skTaxi_BigMLP_ld5_Spectral"))
# 
# train(Settings(G = SmallMLP, D = (s)-> deepcopy(D_dcgan5), epochs = epochs, rand_input = Taxi_input, loss = DCGANLoss(), img_fun = gen_Taxi_images, nclasses = 2, latent_dim = 5, output_dir = "skTaxi_SmallMLP_ld5_Spectral"))

# Gbig, _ = train(Settings(G = BigMLP, D = (s)-> deepcopy(D_dcgan2), epochs = epochs, rand_input = Taxi_input, loss = DCGANLoss(), img_fun = gen_Taxi_images, nclasses = 2, latent_dim = 2, output_dir = "permuted_sk2Taxi_BigMLP_ld2_Spectral_uniformnoise"))
# 
# Gbig = Gbig.net |> cpu
# 
# @save "bigmlp_generator_uniformnoise_permuted.bson" Gbig
# 
# Gsmall, _ = train(Settings(G = SmallMLP, D = (s)-> deepcopy(D_dcgan2), epochs = epochs, rand_input = Taxi_input, loss = DCGANLoss(), img_fun = gen_Taxi_images, nclasses = 2, latent_dim = 2, output_dir = "permuted_sk2Taxi_SmallMLP_ld2_Spectral_uniformnoise"))
# 
# Gsmall = Gsmall.net |> cpu
# 
# @save "smallmlp_generator_uniformnoise_permuted.bson" Gsmall

