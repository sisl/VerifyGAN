using NeuralVerification
using LazySets
using DataStructures
using LinearAlgebra

include("ai2zPQ.jl")
include("tree_utils.jl")

network = read_nnet("../../models/full_mlp_supervised.nnet")

const TOL = Ref(sqrt(eps()))

""" Discretized Ai2z"""

function discretized_ai2z_bounds(network, lbs, ubs, coeffs; n_per_latent = 10, n_per_state = 1)
    # Ai2z overapproximation through discretization 
    ai2z = Ai2z()
    overestimate = -Inf
    underestimate = Inf

    lbs_disc, ubs_disc = get_bounds(lbs, ubs, n_per_latent, n_per_state)

    for (curr_lbs, curr_ubs) in zip(lbs_disc, ubs_disc)
        # Construct the input set, then propagate forwards to a 
        # zonotope over-approximation of the output set
        input_set = Hyperrectangle(low=curr_lbs, high=curr_ubs)
        reach = forward_network(ai2z, network, input_set)

        # The support function ρ maximizes coeffs^T x for x in reach
        curr_overestimate = ρ(coeffs, reach)
        curr_overestimate ≥ overestimate ? overestimate = curr_overestimate : nothing
        # Maximize the negative and take negative to get minimum
        curr_underestimate = -ρ(-coeffs, reach)
        curr_underestimate ≤ underestimate ? underestimate = curr_underestimate : nothing
    end

    return underestimate, overestimate
end

function get_bounds(lbs, ubs, n_per_latent, n_per_state)
    lbs_disc = []
    ubs_disc = []

    for i = 1:n_per_latent
        for j = 1:n_per_latent
            for k = 1:n_per_state
                for l = 1:n_per_state
                    # Find the upper and lower bounds of your region 
                    lb1 = lbs[1] + (i-1)/n_per_latent * (ubs[1] - lbs[1])
                    lb2 = lbs[2] + (j-1)/n_per_latent * (ubs[2] - lbs[2])
                    lb3 = lbs[3] + (k-1)/n_per_state * (ubs[3] - lbs[3])
                    lb4 = lbs[4] + (l-1)/n_per_state * (ubs[4] - lbs[4])
                    ub1 = lbs[1] + (i)/n_per_latent * (ubs[1] - lbs[1])
                    ub2 = lbs[2] + (j)/n_per_latent * (ubs[2] - lbs[2])
                    ub3 = lbs[3] + (k)/n_per_state * (ubs[3] - lbs[3])
                    ub4 = lbs[4] + (l)/n_per_state * (ubs[4] - lbs[4])
                    cur_lbs = [lb1, lb2, lb3, lb4]
                    cur_ubs = [ub1, ub2, ub3, ub4]

                    push!(lbs_disc, cur_lbs)
                    push!(ubs_disc, cur_ubs)
                end
            end
        end
    end

    return lbs_disc, ubs_disc
end

""" ai2zPQ functions """

function ai2zPQ_bounds(network, lbs, ubs, coeffs)
    # Define functions
    evaluate_objective_max(network, x) = dot(coeffs, NeuralVerification.compute_output(network, x))
    evaluate_objective_min(network, x) = dot(-coeffs, NeuralVerification.compute_output(network, x))
    optimize_reach_max(reach) = ρ(coeffs, reach)
    optimize_reach_min(reach) = ρ(-coeffs, reach)

    # Get maximum control
    x, under, value = priority_optimization(network, lbs, ubs, optimize_reach_max, evaluate_objective_max)
    max_control = value
    println(value - under)
    println(x)
    # Get the minimum control
    _, _, value = priority_optimization(network, lbs, ubs, optimize_reach_min, evaluate_objective_min)
    min_control = -value

    return min_control, max_control
end

""" Perform verification on entire tree """

function verify_tree!(tree, network; get_control_bounds = ai2zPQ_bounds, coeffs = [-0.74, -0.44], 
            n_per_latent = 30, n_per_state = 2, latent_bound = 1.0)
    # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, tree.lbs)
    push!(ub_s, tree.ubs)
    push!(s, tree.root_node)

    while !(isempty(s))
        println(length(s))
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        if typeof(curr) == LEAFNODE
            verify_lbs = [-latent_bound; -latent_bound; curr_lbs ./ [6.366468343804353, 17.248858791583547]]
            verify_ubs = [latent_bound; latent_bound; curr_ubs ./ [6.366468343804353, 17.248858791583547]]
            
            min_control, max_control = get_control_bounds(network, verify_lbs, verify_ubs, coeffs)
            
            curr.min_control = min_control
            curr.max_control = max_control
        else
            # Traverse tree and keep track of bounds
            dim = curr.dim
            split = curr.split
            # Go left, upper bounds will change
            left_ubs = copy(curr_ubs)
            left_ubs[dim] = split

            push!(lb_s, curr_lbs)
            push!(ub_s, left_ubs)
            push!(s, curr.left)

            # Go right, lower bounds will change
            right_lbs = copy(curr_lbs)
            right_lbs[dim] = split
            
            push!(lb_s, right_lbs)
            push!(ub_s, curr_ubs)
            push!(s, curr.right)
        end
    end
end