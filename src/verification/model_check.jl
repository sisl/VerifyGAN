using DataStructures
using Distributions

include("tree_utils.jl")
include("../dynamics_model/dubins_model.jl")

function model_check!(tree::KDTREE; max_iterations = 200, belres = 1e-6, γ = 1.0)
    for i = 1:max_iterations
        residual = 0.0
        
        # Stacks to go through tree
        lb_s = Stack{Vector{Float64}}()
        ub_s = Stack{Vector{Float64}}()
        s = Stack{Union{LEAFNODE, KDNODE}}()

        push!(lb_s, tree.lbs)
        push!(ub_s, tree.ubs)
        push!(s, tree.root_node)

        while !isempty(s)
            curr = pop!(s)
            curr_lbs = pop!(lb_s)
            curr_ubs = pop!(ub_s)

            if typeof(curr) == LEAFNODE
                min_cte = curr_lbs[1]
                max_cte = curr_ubs[1]

                if (min_cte < -10.0) || (max_cte > 10.0)
                    curr.prob = 1.0
                else
                    old_prob = curr.prob

                    next_lbs, next_ubs = reachable_cell(curr_lbs, curr_ubs, curr.min_control, curr.max_control)
                    next_nodes = get_overlapping_nodes(tree.root_node, next_lbs, next_ubs)
                    new_prob = γ * maximum([node.prob for node in next_nodes])
                    curr.prob = new_prob

                    change = abs(old_prob - new_prob)
                    change > residual && (residual = change)
                end
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
        println("[Iteration $i] residual: $residual")
        residual < belres && break
    end
end

function forward_reach(init_tree::KDTREE; max_iter = 50, verbose = false)
    # This function assumes that the first tree is labeled already with the states you
    # want to allow the system to start in

    trees = [init_tree]

    for i = 1:max_iter
        tree = trees[end]
        next_tree = copy(init_tree)
        zero_out!(next_tree)
        
        # Stacks to go through tree
        lb_s = Stack{Vector{Float64}}()
        ub_s = Stack{Vector{Float64}}()
        s = Stack{Union{LEAFNODE, KDNODE}}()

        push!(lb_s, tree.lbs)
        push!(ub_s, tree.ubs)
        push!(s, tree.root_node)

        while !isempty(s)
            curr = pop!(s)
            curr_lbs = pop!(lb_s)
            curr_ubs = pop!(ub_s)

            if typeof(curr) == LEAFNODE
                if curr.prob == 1
                    next_lbs, next_ubs = reachable_cell(curr_lbs, curr_ubs, curr.min_control, curr.max_control)
                    #curr_nodes = get_overlapping_nodes(tree.root_node, next_lbs, next_ubs)
                    next_nodes = get_overlapping_nodes(next_tree.root_node, next_lbs, next_ubs)
                    for node in next_nodes
                        node.prob = curr.prob
                    end
                end
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
        push!(trees, next_tree)
        # Check convergence
        converged = tree == next_tree
        verbose && println("[Iteration $i] converged: $converged")
        converged && break
    end
    return trees
end

function forward_reach!(trees::Vector{KDTREE}; verbose = false)
    # This function assumes that the first tree is labeled already with the states you
    # want to allow the system to start in

    for i = 1:length(trees) - 1
        tree = trees[i]
        next_tree = trees[i + 1]
        
        # Stacks to go through tree
        lb_s = Stack{Vector{Float64}}()
        ub_s = Stack{Vector{Float64}}()
        s = Stack{Union{LEAFNODE, KDNODE}}()

        push!(lb_s, tree.lbs)
        push!(ub_s, tree.ubs)
        push!(s, tree.root_node)

        while !isempty(s)
            curr = pop!(s)
            curr_lbs = pop!(lb_s)
            curr_ubs = pop!(ub_s)

            if typeof(curr) == LEAFNODE
                next_lbs, next_ubs = reachable_cell(curr_lbs, curr_ubs, curr.min_control, curr.max_control)
                curr_nodes = get_overlapping_nodes(tree.root_node, next_lbs, next_ubs)
                next_nodes = get_overlapping_nodes(next_tree.root_node, next_lbs, next_ubs)
                for node in next_nodes
                    node.prob = curr.prob
                end
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
        # Check convergence
        converged = tree == next_tree
        verbose && println("[Iteration $i] converged: $converged")
    end
end

function forward_reach(total_tree::KDTREE, init_xy_tree::KDTREE; max_iter = 50, verbose = false)
    # This function assumes that the first tree is labeled already with the states you
    # want to allow the system to start in

    total_tree_next = copy(total_tree)
    zero_out!(total_tree_next)

    label_xy!(init_xy_tree, total_tree)
    trees = [init_xy_tree]

    for i = 1:max_iter        
        # Stacks to go through tree
        lb_s = Stack{Vector{Float64}}()
        ub_s = Stack{Vector{Float64}}()
        s = Stack{Union{LEAFNODE, KDNODE}}()

        push!(lb_s, total_tree.lbs)
        push!(ub_s, total_tree.ubs)
        push!(s, total_tree.root_node)

        while !isempty(s)
            curr = pop!(s)
            curr_lbs = pop!(lb_s)
            curr_ubs = pop!(ub_s)

            if typeof(curr) == LEAFNODE
                if curr.prob == 1.0 # Only need to update from cells that are currently reachable
                    next_lbs, next_ubs = reachable_cell_dtp(curr_lbs, curr_ubs, curr.min_control, curr.max_control)
                    next_nodes = get_overlapping_nodes(total_tree_next.root_node, next_lbs, next_ubs)
                    for node in next_nodes
                        node.prob = curr.prob
                    end
                end
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
        curr_xy_tree = trees[end]
        next_xy_tree = copy(curr_xy_tree)
        label_xy!(next_xy_tree, total_tree_next)
        push!(trees, next_xy_tree)

        copy_labels!(total_tree, total_tree_next)
        zero_out!(total_tree_next)

        verbose && (println("[Iteration:] ", i))
    end
    return trees
end

# NOTE: I actually do not think this works
function forward_reach_runway(total_tree::KDTREE; max_iter = 50, verbose = false)
    # This function assumes that the first tree is labeled already with the states you
    # want to allow the system to start in

    total_tree_next = copy(total_tree)
    zero_out!(total_tree_next)

    init_lbs, init_ubs = get_xy_bounds(total_tree)
    lbs = [init_lbs]
    ubs = [init_ubs]

    for i = 1:max_iter        
        # Stacks to go through tree
        lb_s = Stack{Vector{Float64}}()
        ub_s = Stack{Vector{Float64}}()
        s = Stack{Union{LEAFNODE, KDNODE}}()

        push!(lb_s, total_tree.lbs)
        push!(ub_s, total_tree.ubs)
        push!(s, total_tree.root_node)

        while !isempty(s)
            curr = pop!(s)
            curr_lbs = pop!(lb_s)
            curr_ubs = pop!(ub_s)

            if typeof(curr) == LEAFNODE
                if curr.prob == 1.0 # Only need to update from cells that are currently reachable
                    next_lbs, next_ubs = reachable_cell_dtp(curr_lbs, curr_ubs, curr.min_control, curr.max_control)
                    next_nodes = get_overlapping_nodes(total_tree_next.root_node, next_lbs, next_ubs)
                    for node in next_nodes
                        node.prob = curr.prob
                    end
                end
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
        xy_lbs, xy_ubs = get_xy_bounds(total_tree_next)
        push!(lbs, xy_lbs)
        push!(ubs, xy_ubs)

        copy_labels!(total_tree, total_tree_next)
        zero_out!(total_tree_next)

        verbose && (println("[Iteration:] ", i))
    end
    return lbs, ubs
end

""" Sampling-based Sanity Checks """
function sim_traj(x0, tree::KDTREE; nsteps = 50)
    traj = [x0]
    
    x = copy(x0)
    
    for i = 1:nsteps
        curr_leaf, curr_lbs, curr_ubs = get_leaf_and_bounds(tree, x)
        next_lbs, next_ubs = reachable_cell(curr_lbs, curr_ubs, curr_leaf.min_control, curr_leaf.max_control)
        x = rand.(Uniform.(next_lbs, next_ubs))
        push!(traj, x)
    end
    return traj
end

function sim_traj_true_next(x0, tree::KDTREE; nsteps = 50)
    traj = [x0]
    
    x = copy(x0)
    
    for i = 1:nsteps
        curr_leaf = get_leaf(tree.root_node, x)
        a = rand.(Uniform(curr_leaf.min_control, curr_leaf.max_control))
        xnext, θnext = next_state(x[1], x[2], a)
        x = [xnext, θnext]
        push!(traj, x)
    end
    return traj
end

function find_failures(tree::KDTREE, lbs, ubs; nsteps = 50, npoints = 1000)
    failures = 0
    for i = 1:npoints
        x0 = rand.(Uniform.(lbs, ubs))
        traj = sim_traj(x0, tree, nsteps = nsteps)
        xvals = [traj[i][1] for i = 1:length(traj)]
        if any(xvals .< -10) || any(xvals .> 10)
            failures += 1
        end
    end
    return failures
end

""" Labeling functions """

function label_tree_failures!(tree)
    # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, tree.lbs)
    push!(ub_s, tree.ubs)
    push!(s, tree.root_node)

    while !(isempty(s))
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        if typeof(curr) == LEAFNODE
            # Check if off runway
            min_cte = curr_lbs[1]
            max_cte = curr_ubs[1]

            curr.prob = (min_cte < -10.0) || (max_cte > 10.0) ? 1.0 : 0.0
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

function label_tree_observable!(tree; control_gains = [-0.74, -0.44])
    # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, tree.lbs)
    push!(ub_s, tree.ubs)
    push!(s, tree.root_node)

    while !(isempty(s))
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        if typeof(curr) == LEAFNODE
            curr.min_control = control_gains' * curr_ubs
            curr.max_control = control_gains' * curr_lbs

            # if curr.min_control > curr.max_control
            #     println(curr_ubs)
            #     println(curr_lbs)
            # end
            
            # Check if off runway
            min_cte = curr_lbs[1]
            max_cte = curr_ubs[1]

            curr.prob = (min_cte < -10.0) || (max_cte > 10.0) ? 1.0 : 0.0
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

function label_tree_with_prob!(tree, prob; control_gains = [-0.74, -0.44])
    # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, tree.lbs)
    push!(ub_s, tree.ubs)
    push!(s, tree.root_node)

    while !(isempty(s))
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        if typeof(curr) == LEAFNODE
            curr.min_control = control_gains' * curr_ubs
            curr.max_control = control_gains' * curr_lbs
            
            curr.prob = prob
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

function copy_labels!(copyee, copyer)
    # Stacks to go through tree
    s1 = Stack{Union{LEAFNODE, KDNODE}}()
    s2 = Stack{Union{LEAFNODE, KDNODE}}()

    push!(s1, copyee.root_node)
    push!(s2, copyer.root_node)

    while !(isempty(s1))
        curr_copyee = pop!(s1)
        curr_copyer = pop!(s2)

        if typeof(curr_copyee) == LEAFNODE
            curr_copyee.prob = curr_copyer.prob
        else
            # Traverse trees
            push!(s1, curr_copyee.left)
            push!(s1, curr_copyee.right)
            push!(s2, curr_copyer.left)
            push!(s2, curr_copyer.right)
        end
    end
end

function one_out!(tree)
    # Stacks to go through tree
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(s, tree.root_node)

    while !(isempty(s))
        curr = pop!(s)

        if typeof(curr) == LEAFNODE
            curr.prob = 1.0
        else
            # Traverse tree
            push!(s, curr.left)
            push!(s, curr.right)
        end
    end
end

function zero_out!(tree)
    # Stacks to go through tree
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(s, tree.root_node)

    while !(isempty(s))
        curr = pop!(s)

        if typeof(curr) == LEAFNODE
            curr.prob = 0.0
        else
            # Traverse tree
            push!(s, curr.left)
            push!(s, curr.right)
        end
    end
end

function label_xy!(xy_tree, total_tree)
    # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, xy_tree.lbs)
    push!(ub_s, xy_tree.ubs)
    push!(s, xy_tree.root_node)

    while !(isempty(s))
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        if typeof(curr) == LEAFNODE
            total_lbs = [curr_lbs[1]; total_tree.lbs[2]; curr_lbs[2]]
            total_ubs = [curr_ubs[1]; total_tree.ubs[2]; curr_ubs[2]]
            nodes = get_overlapping_nodes(total_tree.root_node, total_lbs, total_ubs)

            max_prob = maximum([node.prob for node in nodes])
            if max_prob == 1.0
                curr.prob = 1.0
            end
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

# Modifies control outputs
function label_xyθ!(total_tree, xθ_tree)
    # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, total_tree.lbs)
    push!(ub_s, total_tree.ubs)
    push!(s, total_tree.root_node)

    while !(isempty(s))
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        if typeof(curr) == LEAFNODE
            point = (curr_ubs[1:2] .+ curr_lbs[1:2]) ./ 2
            leaf = get_leaf(xθ_tree.root_node, point)
            curr.min_control = leaf.min_control
            curr.max_control = leaf.max_control
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

function label_start_states!(tree, lbs, ubs)
    zero_out!(tree)
    nodes = get_overlapping_nodes(tree.root_node, lbs, ubs)
    for node in nodes
        node.prob = 1.0
    end
end

function get_xy_bounds(total_tree)
     # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, total_tree.lbs)
    push!(ub_s, total_tree.ubs)
    push!(s, total_tree.root_node)

    lbs = total_tree.ubs[[1, 3]]
    ubs = total_tree.lbs[[1, 3]]

    while !(isempty(s))
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        if typeof(curr) == LEAFNODE
            if curr.prob == 1.0
                xy_lbs = curr_lbs[[1, 3]]
                xy_ubs = curr_ubs[[1, 3]]
                
                replace_lb_dims = findall(xy_lbs .< lbs)
                replace_ub_dims = findall(xy_ubs .> ubs)

                if length(replace_lb_dims) > 0
                    lbs[replace_lb_dims] = xy_lbs[replace_lb_dims]
                end

                if length(replace_ub_dims) > 0
                    ubs[replace_ub_dims] = xy_ubs[replace_ub_dims]
                end
            end
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

    return lbs, ubs
end