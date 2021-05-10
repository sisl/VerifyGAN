using Flux
using BSON
using LinearAlgebra
using Distributions
using HDF5

function get_approximate_minimum_radius(g, lbs, ubs, y₀; num_samples = 1000, p = 2)
    sample_inputs = get_sample(lbs, ubs, num_samples)
    images = g(sample_inputs)
    images = reshape(images, 128, :)

    #target = (y₀ .* 2) .- 1

    dists = [norm(images[:, i] - y₀, p) for i = 1:num_samples]
    ind = argmin(dists)

    return sample_inputs[:, ind], dists[ind]
end

function get_sample(lbs, ubs, num_samples)
    dist = Uniform.(lbs, ubs)
    samples = hcat(rand.(dist, num_samples)...)
    return Float32.(samples')
end

function get_approximate_minimum_radii(g, real_images, labels; state_eps = 1e-5, p = 2, latent_bound = 1.0)
    n = size(real_images, 2)
    xs = zeros(4, n)
    overs = zeros(n)

    for i = 1:n
        i % 10 == 0 ? println(i) : nothing
        
        lbs = [-latent_bound, -latent_bound, labels[1, i] / 6.366468343804353 - state_eps, 
                                    labels[2, i] / 17.248858791583547 - state_eps]
        ubs = [latent_bound, latent_bound, labels[1, i] / 6.366468343804353 + state_eps, 
                                    labels[2, i] / 17.248858791583547 + state_eps]

        y₀ = (real_images[:, i] .* 2) .- 1;

        x, over = get_approximate_minimum_radius(g, lbs, ubs, y₀, p = p)
        xs[:, i] = x
        overs[i] = over
    end

    return xs, overs
end

g = BSON.load("../gan_training/generators/supervised_mlp.bson")[:g]

fn = "../../data/SK_DownsampledGANFocusAreaData.h5"
images = h5read(fn, "y_train")
images = reshape(images, 16*8, :)
y = h5read(fn, "X_train")[1:2, :]

@time xs, overs = get_approximate_minimum_radii(g, images, y, p = 2, latent_bound = 0.8)