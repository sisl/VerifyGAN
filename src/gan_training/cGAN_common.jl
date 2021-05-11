using CUDA, Flux, MLDatasets, Statistics, Images, Parameters, Printf, Random
using Base.Iterators: partition
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using Flux.Data: DataLoader
using Flux.Optimise: train!, @epochs
using Parameters, Printf, Random
using BSON: @save
using Distributions
using Zygote

include("spectral_norm.jl")


@with_kw struct Settings
	G = nothing
	D = nothing
	loss = nothing
	img_fun = nothing
	rand_input = nothing
	batch_size::Int = 128
	latent_dim::Int = 100
	nclasses::Int = 2
	epochs::Int = 120
	verbose_freq::Int = 5
	output_x::Int = 6
	output_y::Int = 6 
	optD = ADAM(0.0002, (0.5, 0.99))
	optG = ADAM(0.0002, (0.5, 0.99))
	output_dir = "output" # save dir
end

## Regular DCGAN
struct DCGANLoss end

function Lᴰ(t::DCGANLoss, G, D, z, ny, x, y)
	real_loss = logitbinarycrossentropy(D(x, y), 1f0, agg=mean)
	fake_loss = logitbinarycrossentropy(D(G(z, ny), ny), 0f0, agg=mean)
	return real_loss + fake_loss
end

Lᴳ(t::DCGANLoss, G, D, z, ny) = logitbinarycrossentropy(D(G(z, ny), ny), 1f0, agg=mean)

## Least Squares Loss
struct LSLoss end

function Lᴰ(t::LSLoss, G, D, z, ny, x, y)
	Flux.mse(D(x, y), 1f0) + Flux.mse(D(G(z, ny), ny), 0f0)
end

Lᴳ(t::LSLoss, G, D, z, ny) = Flux.mse(D(G(z, ny), ny), 1f0)

## WGAN-GP
@with_kw struct WLossGP
	λ = 10f0 # gp penalty
	gp_type = :exact #:approx or :exact
	noise_range = 1 # Range of the ϵ parameter that picks a point to evalutate the gradient (smaller is closer to "real" data)
end

function gradient_penalty(D, x, y)
	B = size(y, 2)
	l, b = Flux.pullback(() -> D(x, y), Flux.params(x, y))
	grads = b(ones(Float32, 1, size(x, 4)) |> gpu)
	Flux.mean((sqrt.(sum(reshape(grads[x], :, B).^2, dims = 1) .+ sum(grads[y].^2, dims = 1)) .- 1f0).^2)
end

function approx_penalty(D, x, xhat, y, yhat)
	B = size(x, 4) # batch size
	Δx, Δy = reshape(x .- xhat, :, B), y .- yhat # difference
	mag = sqrt.(sum(Δx.^2, dims = 1) + sum(Δy.^2, dims = 1))
	xdir, ydir = reshape(Δx ./ mag, size(xhat)), Δy ./ mag
	δ = 0.01f0
	ΔD = abs.(D(xhat, yhat) .- D(xhat .+ δ .* xdir, yhat .+ δ .* ydir))
	Flux.mean((ΔD ./ δ .- 1f0).^2)
	Flux.mean(max.((ΔD ./ δ .- 1f0), 0f0).^2)
end

function Lᴰ(t::WLossGP, G, D, z, ny, x, y)
	# ϵ = Zygote.ignore(() -> Float32.(rand(Uniform(0, t.noise_range), 1, size(y,2))) |> gpu)
	xtilde = G(z, ny)
	# ϵx = reshape(ϵ, 1, 1, 1, length(ϵ))
	# xhat = ϵx .* xtilde + (1f0 .- ϵx) .* x
	# yhat = ϵ .* ny + (1f0 .- ϵ) .* y
	loss = mean(D(xtilde, ny) .- D(x, y))
	# if t.gp_type == :approx
	# 	loss += t.λ*approx_penalty(D, x, xhat, y, yhat)
	# else
	# 	loss += t.λ*gradient_penalty(D, xhat, yhat)
	# end
end

Lᴳ(t::WLossGP, G, D, z, ny) = -mean(D(G(z, ny), ny)) 

## hinge
struct HingeLoss end

function Lᴰ(t::HingeLoss, G, D, z, ny, x, y)
	real_loss = mean(relu.(1f0 .- D(x, y)))
	fake_loss = mean(relu.(1f0 .+ D(G(z, ny), ny)))
	return real_loss + fake_loss
end

Lᴳ(t::HingeLoss, G, D, z, ny) = -mean(D(G(z, ny), ny)) 



## Training Code
function train_discriminator!(loss, G, D, z, ny, x, y, optD)
    θ = Flux.params(D)
    loss, back = Flux.pullback(() -> Lᴰ(loss, G, D, z, ny, x, y), θ)
    update!(optD, θ, back(1f0))
    loss
end

function orthogonal_regularization(G, β=1f-4)
    reg = 0f0
    for l in Zygote.ignore(()->filter((l)->hasproperty(l, :weight), vcat([[l...] for l in Flux.trainable(G)]...)))
		W = reshape(l.weight, :, size(l.weight)[end])
        prod = W' * W
        mat = Zygote.ignore(() -> ones(Float32, size(prod)...) .- Matrix{Float32}(I, size(prod)...) |> gpu)
        reg += norm(prod .* mat)^2
    end
    β*reg
end


function train_generator!(loss, G, D, z, ny, optG)
	θ = Flux.params(G)
	loss, back = Flux.pullback(() -> Lᴳ(loss, G, D, z, ny) + orthogonal_regularization(G), θ)
	update!(optG, θ, back(1f0))
	loss
end

function to_image(G, fixed_noise, fixed_labels, s)
    fake_images = cpu.(G.(fixed_noise, fixed_labels))
    image_array = permutedims(dropdims(reduce(vcat, reduce.(hcat, partition(fake_images, s.output_y))); dims=(3, 4)), (2, 1))
    image_array = map(clamp01nan, Gray.(image_array .+ 1f0) ./ 2f0)
    return image_array
end

function train(s::Settings)
	G, D = s.G(s), s.D(s) # Build the models
	data, fixed_noise, fixed_labels = s.img_fun(s) # Load the data
	
    # Training
	Dhist, Ghist = [], []
	for e=1:s.epochs 
		@info("Epoch $(e)!")
		avg_loss_D, avg_loss_G, steps = 0, 0, 0
		for (x, y) in data
			loss_D = train_discriminator!(s.loss, G, D, s.rand_input(s)..., x, y, s.optD)
			loss_G = train_generator!(s.loss, G, D, s.rand_input(s)..., s.optG)
			
			push!(Dhist, loss_D)
			push!(Ghist, loss_G)
			avg_loss_D += loss_D
			avg_loss_G += loss_G
			steps += 1			
		end
		avg_loss_D /= steps
		avg_loss_G /= steps
		if e % s.verbose_freq == 0
			@info("Discriminator loss = $avg_loss_D, Generator loss = $avg_loss_G)")
			name = @sprintf("cgan_epochs_%06d.png", e)
			save(string(s.output_dir, "/", name), to_image(G, fixed_noise, fixed_labels, s))
		end
    end
	G, D, Ghist, Dhist
end 

