using PGFPlots
using Colors
using ColorBrewer

function plot_estimated_prob(tree)
    nbin = 200
    xmin = tree.lbs[1]
    xmax = tree.ubs[1]
    ymin = tree.lbs[2]
    ymax = tree.ubs[2]

    #safe_color = RGB(0.0, 0.0, 1.0) # blue
    safe_color = RGB(169.0 / 255.0, 169.0 / 255.0, 169.0 / 255.0) # gray
    #safe_color = RGB(105.0 / 255.0, 105.0 / 255.0, 105.0 / 255.0) # dark gray
    #unsafe_color = RGB(1.0, 0.0, 0.0) # red
    unsafe_color = RGB(139.0 / 255.0, 0.0, 0.0) # dark red

    colors = [safe_color, unsafe_color]

    ax = Axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width="7cm", height="8cm", 
    ylabel=L"$\theta$ (degrees)", xlabel=L"$x$ (meters)", title="Safe Cells")

    function get_heat(x, y)
        traj = sim_traj_true_next([x, y], tree)
        xvals = [traj[i][1] for i = 1:length(traj)]
        prob = any(xvals .< -10) || any(xvals .> 10)
        return prob
    end

    push!(ax, Plots.Image(get_heat, (xmin, xmax), (ymin, ymax), zmin = 0, zmax = 1,
        xbins = nbin, ybins = nbin, colormap = ColorMaps.RGBArrayMap(colors), colorbar=false))

    return ax
end

function plot_prob(tree)
    nbin = 200
    xmin = tree.lbs[1]
    xmax = tree.ubs[1]
    ymin = tree.lbs[2]
    ymax = tree.ubs[2]

    #safe_color = RGB(0.0, 0.0, 1.0) # blue
    safe_color = RGB(169.0 / 255.0, 169.0 / 255.0, 169.0 / 255.0) # gray
    #safe_color = RGB(105.0 / 255.0, 105.0 / 255.0, 105.0 / 255.0) # dark gray
    #unsafe_color = RGB(1.0, 0.0, 0.0) # red
    unsafe_color = RGB(139.0 / 255.0, 0.0, 0.0) # dark red

    colors = [safe_color, unsafe_color]

    ax = Axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width="7cm", height="8cm", 
    ylabel=L"$\theta$ (degrees)", xlabel=L"$x$ (meters)", title="Safe Cells")

    function get_heat(x, y)
        leaf = get_leaf(tree.root_node, [x, y])
        return leaf.prob
    end

    push!(ax, Plots.Image(get_heat, (xmin, xmax), (ymin, ymax), zmin = 0, zmax = 1,
        xbins = nbin, ybins = nbin, colormap = ColorMaps.RGBArrayMap(colors), colorbar=false))

    return ax
end

function plot_control_range(tree; nbin = 200)
    xmin = tree.lbs[1]
    xmax = tree.ubs[1]
    ymin = tree.lbs[2]
    ymax = tree.ubs[2]


    ax = Axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width="7cm", height="8cm", 
    ylabel=L"$\theta$ (degrees)", xlabel=L"$x$ (meters)", title="Control Range")

    function get_heat(x, y)
        leaf = get_leaf(tree.root_node, [x, y])
        return leaf.max_control - leaf.min_control
    end

    push!(ax, Plots.Image(get_heat, (xmin, xmax), (ymin, ymax),
        xbins = nbin, ybins = nbin, colormap = pasteljet, colorbar=false))

    return ax
end

function plot_reachable(tree)
    nbin = 200
    xmin = tree.lbs[1]
    xmax = tree.ubs[1]
    ymin = tree.lbs[2]
    ymax = tree.ubs[2]

    unreach_color = RGB(1.0, 1.0, 1.0) # white
    #reach_color = RGB(0.0, 0.0, 0.502) # navy blue
    reach_color = RGB(0.0, 0.5, 0.5) # teal

    colors = [unreach_color, reach_color]

    ax = Axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width="7cm", height="8cm", 
    ylabel=L"$\theta$ (degrees)", xlabel=L"$x$ (meters)", title="Reachable Cells")

    function get_heat(x, y)
        leaf = get_leaf(tree.root_node, [x, y])
        return leaf.prob
    end

    push!(ax, Plots.Image(get_heat, (xmin, xmax), (ymin, ymax), zmin = 0, zmax = 1,
        xbins = nbin, ybins = nbin, colormap = ColorMaps.RGBArrayMap(colors), colorbar=false))

    return ax
end

function plot_cells(tree; color = "blue", opacity = 1.0, lw = 1)
    xmin = tree.lbs[1]
    xmax = tree.ubs[1]
    ymin = tree.lbs[2]
    ymax = tree.ubs[2]

    #safe_color = RGB(0.0, 0.0, 1.0) # blue
    safe_color = RGB(169.0 / 255.0, 169.0 / 255.0, 169.0 / 255.0) # gray
    #safe_color = RGB(105.0 / 255.0, 105.0 / 255.0, 105.0 / 255.0) # dark gray
    #unsafe_color = RGB(1.0, 0.0, 0.0) # red
    unsafe_color = RGB(139.0 / 255.0, 0.0, 0.0) # dark red

    colors = [safe_color, unsafe_color]

    ax = Axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width="7cm", height="8cm", 
    ylabel=L"$\theta$ (degrees)", xlabel=L"$x$ (meters)")

    lbs, ubs, probs = get_bounds_and_probs(tree)

    for i = 1:length(lbs)
        if probs[i] == 1.0
            #push!(ax, Plots.Command(get_filled_rectangle_border(lbs[i], ubs[i], color, 
                            #opacity = opacity, lw = lw)))
            push!(ax, Plots.Command(get_rectangle(lbs[i], ubs[i], color, lw)))
        else
            push!(ax, Plots.Command(get_rectangle(lbs[i], ubs[i], "black", 0.1)))
        end
    end

    return ax
end

function plot_reachable_xy_cells(xy_tree; color = "blue", opacity = 1.0)
    nbin = 200
    xmin = xy_tree.lbs[1]
    xmax = xy_tree.ubs[1]
    ymin = xy_tree.lbs[2]
    ymax = xy_tree.ubs[2]

    ax = Axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width="7cm", height="8cm", 
    ylabel=L"$y$ (meters)", xlabel=L"$x$ (meters)", title="Reachable Cells")

    lbs, ubs, probs = get_bounds_and_probs(xy_tree)

    for i = 1:length(lbs)
        if probs[i] == 1.0
            push!(ax, Plots.Command(get_filled_rectangle(lbs[i], ubs[i], color, opacity = opacity)))
        end
    end

    return ax
end

function get_road(;xmin=0.0, xmax=150.0, ymin=-12.0, ymax=12.0)
	ax = Axis(PGFPlots.Plots.Linear([xmin, xmax], [-10.0, -10.0], mark="none", 
            style="black,solid,thick, name path=A"))
	push!(ax, PGFPlots.Plots.Linear([xmin, xmax], [10.0, 10.0], mark="none", 
            style="black,solid,thick, name path=B"))
	push!(ax, PGFPlots.Plots.Linear([xmin, xmax], [0.0, 0.0], mark="none", 
            style="white,dashed,thick"))
	push!(ax, PGFPlots.Plots.Command("\\path[name path=axis] (axis cs:$xmin,$ymin) -- (axis cs:$xmax,$ymin);"))
	push!(ax, PGFPlots.Plots.Command("\\path[name path=axisright] (axis cs:$xmin,$ymax) -- (axis cs:$xmax,$ymax);"))
	push!(ax, PGFPlots.Plots.Command("\\addplot[pastelGreen!40] fill between[of=A and axis];"))
	push!(ax, PGFPlots.Plots.Command("\\addplot[pastelGreen!40] fill between[of=B and axisright];"))
	push!(ax, PGFPlots.Plots.Command("\\addplot[black!40] fill between[of=A and B];"))
	ax.width = "15cm"
	ax.height = "4cm"
    ax.xmin = xmin
    ax.xmax = xmax
    ax.ymin = ymin
    ax.ymax = ymax
    ax.xlabel = "Downtrack Position (m)"
    ax.ylabel = "Crosstrack Error (m)"
	#ax.axisEqualImage = true
	return ax
end

function plot_reachable_cells_on_road(xy_tree::KDTREE; color = "blue", opacity = 1.0, 
                                xmin=0.0, xmax=150.0, ymin=-12.0, ymax=12.0)
    ax = get_road(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)
    
    lbs, ubs, probs = get_bounds_and_probs(xy_tree)

    for i = 1:length(lbs)
        if probs[i] == 1.0
            push!(ax, Plots.Command(get_filled_rectangle(reverse(lbs[i]), reverse(ubs[i]),
                                                            color, opacity = opacity)))
        end
    end

    return ax
end

function plot_reachable_cells_on_road(lbs, ubs; color = "blue", opacity = 1.0, 
                                xmin=0.0, xmax=150.0, ymin=-12.0, ymax=12.0)
    ax = get_road(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)

    for i = 1:length(lbs)
        push!(ax, Plots.Command(get_filled_rectangle(reverse(lbs[i]), reverse(ubs[i]),
                                                        color, opacity = opacity)))
    end

    return ax
end
                

function plot_reachable_cells_and_trajs_on_road(xy_tree, xres, yres; color = "blue", opacity = 1.0, 
                                xmin=0.0, xmax=150.0, ymin=-12.0, ymax=12.0)
    ax = get_road(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)
    for i = 1:length(xres)
        push!(ax, Plots.Linear(yres[i][1:5:end], xres[i][1:5:end], mark = "none", style = "solid, blue"))
    end

    lbs, ubs, probs = get_bounds_and_probs(xy_tree)

    for i = 1:length(lbs)
        if probs[i] == 1.0
            push!(ax, Plots.Command(get_filled_rectangle(reverse(lbs[i]), reverse(ubs[i]),
                                                            color, opacity = opacity)))
        end
    end

    return ax
end

""" Plotting Utils """

function get_filled_rectangle(lb, ub, color; opacity = 1.0)
    return "\\fill [$(color), opacity=$(opacity)] (axis cs:$(string(lb[1])),$(string(lb[2]))) rectangle (axis cs:$(string(ub[1])),$(string(ub[2])));"
end

function get_filled_rectangle_border(lb, ub, color; opacity = 1.0, lw = 1)
    return "\\draw [fill=$(color), $(color), opacity=$(opacity), line width = $(lw)] (axis cs:$(string(lb[1])),$(string(lb[2]))) rectangle (axis cs:$(string(ub[1])),$(string(ub[2])));"
end

function get_rectangle(lb, ub)
    return "\\draw ($(string(lb[1])),$(string(lb[2]))) rectangle ($(string(ub[1])),$(string(ub[2])));"
end

function get_rectangle(lb, ub, color, lw)
    return "\\draw [$(color), line width = $(lw)] ($(string(lb[1])),$(string(lb[2]))) rectangle ($(string(ub[1])),$(string(ub[2])));"
end