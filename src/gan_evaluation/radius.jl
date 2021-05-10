using HDF5
using Convex
using Mosek, MosekTools

include("../verification/ai2zPQ.jl");

function dist_to_zonotope_p(reach, point; p = 2)
    G = reach.generators
    c = reach.center
    n, m = size(G)
    x = Variable(m)
    obj = norm(G * x + c - point, p)
    prob = minimize(obj, [x <= 1.0, x >= -1.0])
    solve!(prob, Mosek.Optimizer(LOG=0, MAX_NUM_WARNINGS=0))
    #@assert prob.status == OPTIMAL "Solve must result in optimal status"
    return prob.optval
end

function get_minimum_radius(network, lbs, ubs, y₀; p = 2)
    evaluate_objective(network, x) = -norm(y₀ - NeuralVerification.compute_output(network, x), p)
    optimize_reach(reach) = -dist_to_zonotope_p(reach, y₀; p=p)
    return priority_optimization(network, lbs, ubs, optimize_reach, evaluate_objective)
end

function get_minimum_radii(network, real_images, labels; state_eps = 1e-5, p = 2)
    n = size(real_images, 2)
    xs = zeros(4, n)
    unders = zeros(n)
    overs = zeros(n)

    for i = 1:n
        i % 10 == 0 ? println(i) : nothing
        
        lbs = [-0.8, -0.8, labels[1, i] / 6.366468343804353 - state_eps, 
                                    labels[2, i] / 17.248858791583547 - state_eps]
        ubs = [0.8, 0.8, labels[1, i] / 6.366468343804353 + state_eps, 
                                    labels[2, i] / 17.248858791583547 + state_eps]

        y₀ = (real_images[:, i] .* 2) .- 1;

        x, under, over = get_minimum_radius(network, lbs, ubs, y₀, p = p)
        xs[:, i] = x
        unders[i] = -over
        overs[i] = -under
    end

    return xs, unders, overs
end

network = read_nnet("../gan_training/generators/supervised_mlp.nnet");

fn = "../../data/SK_DownsampledGANFocusAreaData.h5"
images = h5read(fn, "y_train")
images = reshape(images, 16*8, :)
y = h5read(fn, "X_train")[1:2, :]

@time xs, unders, overs = get_minimum_radii(network, images, y, p = 2)