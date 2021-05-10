using DataStructures
using LazySets

mutable struct LEAFNODE
    min_control::Float64
    max_control::Float64
    prob::Float64
end

mutable struct KDNODE
    dim::Int32
    split::Float64
    left::Union{KDNODE, LEAFNODE}
    right::Union{KDNODE, LEAFNODE}
    min_control::Float64
    max_control::Float64
end

mutable struct KDTREE
    lbs::Vector{Float64}
    ubs::Vector{Float64}
    root_node::KDNODE
end

function leafnode(;min_control = -10.0, max_control = 10.0, prob = 0.0)
    return LEAFNODE(min_control, max_control, prob)
end

function kdnode(;dim = 1, split = 0.0, left = leafnode(), right = leafnode(), 
    min_control = -10.0, max_control = 10.0)
    return KDNODE(dim, split, left, right, min_control, max_control)
end

function get_leaf(root_node, point)
    curr_node = root_node
    while typeof(curr_node) != LEAFNODE
        val = point[curr_node.dim]
        split = curr_node.split
        curr_node = val < split ? curr_node.left : curr_node.right
    end
    return curr_node
end

function get_bounds(tree)
    lbs = []
    ubs = []
    
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
            push!(lbs, curr_lbs)
            push!(ubs, curr_ubs)
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

function get_leaf_and_bounds(tree, point)
    curr_node = tree.root_node
    lbs = copy(tree.lbs)
    ubs = copy(tree.ubs)
    while typeof(curr_node) != LEAFNODE
        dim = curr_node.dim
        val = point[dim]
        split = curr_node.split
        if val < split
        	curr_node = curr_node.left
        	ubs[dim] = split
        else
        	curr_node = curr_node.right
        	lbs[dim] = split
        end
    end
    return curr_node, lbs, ubs
end

function get_bounds_and_probs(tree)
    lbs = []
    ubs = []
    probs = []
    
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
            push!(lbs, curr_lbs)
            push!(ubs, curr_ubs)
            push!(probs, curr.prob)
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

    return lbs, ubs, probs
end

function get_overlapping_nodes(root_node, lbs, ubs)
    curr_node_list = Vector{LEAFNODE}()
    overlapping_nodes_helper(root_node, lbs, ubs, curr_node_list)
    return curr_node_list
end

function overlapping_nodes_helper(curr_node, lbs, ubs, curr_node_list)
    # Base case
    if typeof(curr_node) == LEAFNODE
        push!(curr_node_list, curr_node)
    else
        # Check if can prune either side
        dim = curr_node.dim
        split = curr_node.split
        if split > ubs[dim]
            # Can prune the right half
            overlapping_nodes_helper(curr_node.left, lbs, ubs, curr_node_list)
        elseif split < lbs[dim]
            # Can prune the left half
            overlapping_nodes_helper(curr_node.right, lbs, ubs, curr_node_list)
        else
            overlapping_nodes_helper(curr_node.left, lbs, ubs, curr_node_list)
            overlapping_nodes_helper(curr_node.right, lbs, ubs, curr_node_list)
        end
    end
end

function split!(s, lb_s, ub_s, curr_node, curr_lbs, curr_ubs, dim)
    split = (curr_lbs[dim] + curr_ubs[dim]) / 2

    curr_node.split = split
    curr_node.dim = dim

    curr_node.left = kdnode(min_control = curr_node.min_control, max_control = curr_node.max_control)
    curr_node.right = kdnode(min_control = curr_node.min_control, max_control = curr_node.max_control)

    # Get new bounds
    # Go left, upper bounds will change
    left_lbs = copy(curr_lbs)
    left_ubs = copy(curr_ubs)
    left_ubs[dim] = split
    # Go right, lower bounds will change
    right_lbs = copy(curr_lbs)
    right_lbs[dim] = split
    right_ubs = copy(curr_ubs)

    # Add everything to the stacks
    push!(s, curr_node.left)
    push!(lb_s, curr_lbs)
    push!(ub_s, left_ubs)
    push!(s, curr_node.right) 
    push!(lb_s, right_lbs)
    push!(ub_s, curr_ubs)
end

function split_specific_dims!(s, lb_s, ub_s, curr_node, curr_lbs, curr_ubs, dims)
    split!(s, lb_s, ub_s, curr_node, curr_lbs, curr_ubs, dims[1])
    for i = 2:length(dims)
        nodes = []
        lbs = []
        ubs = []
        for j = 1:2^(i-1)
            push!(nodes, pop!(s))
            push!(lbs, pop!(lb_s))
            push!(ubs, pop!(ub_s))
        end
        for j = 1:2^(i-1)
            split!(s, lb_s, ub_s, nodes[j], lbs[j], ubs[j], dims[i])
        end
    end
end

function create_tree(min_widths, lbs, ubs)
    root_node = kdnode()
    
    # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, lbs)
    push!(ub_s, ubs)
    push!(s, root_node)

    while !isempty(s)
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        # println("lbs: ", curr_lbs)
        # println("ubs: ", curr_ubs)

        # Figure out which dimensions need splitting (if any)
        curr_widths = curr_ubs .- curr_lbs
        # println("widths: ", curr_widths)
        # println()
        dims_to_split = findall(curr_widths .> min_widths)

        # Split them
        if length(dims_to_split) > 0
            split_specific_dims!(s, lb_s, ub_s, curr, curr_lbs, curr_ubs, dims_to_split)
        end
    end

    # Traverse and remove extra nodes (this is pretty hacky)
    s = Stack{Union{LEAFNODE, KDNODE}}()
    push!(s, root_node)

    while !isempty(s)
        curr = pop!(s)
        
        if typeof(curr.left.left) == LEAFNODE
            curr.left = leafnode()
        else
            push!(s, curr.left)
        end

        if typeof(curr.right.right) == LEAFNODE
            curr.right = leafnode()
        else
            push!(s, curr.right)
        end
    end

    return KDTREE(lbs, ubs, root_node)
end

function Base.:(==)(k1::KDNODE, k2::KDNODE)
    return k1.dim == k2.dim && k1.split == k2.split && k1.left == k2.left && k1.right == k2.right && 
                k1.min_control == k2.min_control && k1.max_control == k2.max_control
end

function Base.:(==)(l1::LEAFNODE, l2::LEAFNODE)
    return l1.min_control == l2.min_control && l1.max_control == l2.max_control && l1.prob == l2.prob
end


function Base.:(==)(t1::KDTREE, t2::KDTREE)
    if (t1.lbs != t2.lbs) || (t1.ubs != t2.ubs)
        return false
    else
        equal = true

        s1 = Stack{Union{LEAFNODE, KDNODE}}()
        s2 = Stack{Union{LEAFNODE, KDNODE}}()

        push!(s1, t1.root_node)
        push!(s2, t2.root_node)

        while !isempty(s1)
            curr1 = pop!(s1)
            curr2 = pop!(s2)

            equal = curr1 == curr2

            if !equal
                return false
            else
                if typeof(curr1) != LEAFNODE
                    push!(s1, curr1.left)
                    push!(s1, curr1.right)
                    push!(s2, curr2.left)
                    push!(s2, curr2.right)
                end
            end
        end
    end
    return equal
end

function Base.copy(l::LEAFNODE)
    return LEAFNODE(l.min_control, l.max_control, l.prob)
end

function Base.copy(tree::KDTREE)
    root = tree.root_node
    if typeof(root) == LEAFNODE
        new_tree = copy(root)
    else
        new_tree = kdnode(dim = root.dim, split = root.split, 
                                min_control = root.min_control, max_control = root.max_control)
        
        s_old = Stack{Union{LEAFNODE, KDNODE}}()
        s_new = Stack{Union{LEAFNODE, KDNODE}}()

        push!(s_old, tree.root_node)
        push!(s_new, new_tree)

        while !isempty(s_old)
            curr_old = pop!(s_old)
            curr_new = pop!(s_new)

            old_left = curr_old.left
            old_right = curr_old.right

            left_leaf = typeof(old_left) == LEAFNODE
            right_leaf = typeof(old_right) == LEAFNODE

            if left_leaf && right_leaf
                curr_new.left = copy(old_left)
                curr_new.right = copy(old_right)
            elseif left_leaf
                curr_new.left = copy(old_left)
                curr_new.right = kdnode(dim = old_right.dim, split = old_right.split, 
                                min_control = old_right.min_control, max_control = old_right.max_control)
                push!(s_old, old_right)
                push!(s_new, curr_new.right)
            elseif right_leaf
                curr_new.right = copy(old_right)
                curr_new.left = kdnode(dim = old_left.dim, split = old_left.split, 
                                min_control = old_left.min_control, max_control = old_left.max_control)
                push!(s_old, old_left)
                push!(s_new, curr_new.left)
            else
                curr_new.left = kdnode(dim = old_left.dim, split = old_left.split, 
                                min_control = old_left.min_control, max_control = old_left.max_control)
                push!(s_old, old_left)
                push!(s_new, curr_new.left)
                curr_new.right = kdnode(dim = old_right.dim, split = old_right.split, 
                                min_control = old_right.min_control, max_control = old_right.max_control)
                push!(s_old, old_right)
                push!(s_new, curr_new.right)
            end
        end
    end
    return KDTREE(copy(tree.lbs), copy(tree.ubs), new_tree)
end