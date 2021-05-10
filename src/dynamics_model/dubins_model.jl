# Parameters

L = 5 # m
v = 5 # m/s

# Dynamics model functions

function next_state(x, θ, ϕ; dt = 0.05)
    # x is in meters and θ is in degrees
	# ϕ is steering angle input (deg)
	
	# Dynamics model
	ẋ = v * sind(θ)
	θ̇ = (v / L) * tand(ϕ)
	
	x′ = x + ẋ * dt
	θ′ = θ + rad2deg(θ̇) * dt
	
	return x′, θ′
end

function next_state(x, θ, y, ϕ; dt = 0.05)
    # for plotting purposes, we might also want the y value
    # x is in meters and θ is in degrees
	# ϕ is steering angle input (deg)
	
	# Dynamics model
	ẋ = v * sind(θ)
	ẏ = v * cosd(θ)
	θ̇ = (v / L) * tand(ϕ)
	
	x′ = x + ẋ * dt
	y′ = y + ẏ * dt
	θ′ = θ + rad2deg(θ̇) * dt
	
	return x′, θ′, y′
end

function next_state(x, θ, y, ϕ, cosval; dt = 0.05)
    # for plotting purposes, we might also want the y value
    # x is in meters and θ is in degrees
	# ϕ is steering angle input (deg)
	
	# Dynamics model
	ẋ = v * sind(θ)
	ẏ = v * cosval
	θ̇ = (v / L) * tand(ϕ)
	
	x′ = x + ẋ * dt
	y′ = y + ẏ * dt
	θ′ = θ + rad2deg(θ̇) * dt
	
	return x′, θ′, y′
end

function sim_steps(x, θ, ϕ; dt = 0.05, num_steps = 20)
    for i = 1:num_steps
		x, θ = next_state(x, θ, ϕ)
	end
    return x, θ
end

function sim_steps(x, θ, y, ϕ; dt = 0.05, num_steps = 20)
    for i = 1:num_steps
		x, θ, y = next_state(x, θ, y, ϕ)
	end
    return x, θ, y
end

function sim_steps(x, θ, y, ϕ, cosval; dt = 0.05, num_steps = 20)
    for i = 1:num_steps
		x, θ, y = next_state(x, θ, y, ϕ)
	end
    return x, θ, y
end

# NOTE: this function is only valid for θs between -90 and 90 degrees
function reachable_cell(lbs, ubs, ϕ_min, ϕ_max)
	next_lb_x, next_lb_θ = sim_steps(lbs[1], lbs[2], ϕ_min)
	next_ub_x, next_ub_θ = sim_steps(ubs[1], ubs[2], ϕ_max)
	return [next_lb_x, next_lb_θ], [next_ub_x, next_ub_θ]
end

function cosdextrema(xmin, xmax)
	b1, b2 = cosd(xmin), cosd(xmax)
	curr_min, curr_max = min(b1, b2), max(b1, b2)
	if xmin < 0 && 0 < xmax
		curr_max = 1
	end
	curr_min, curr_max
end

function reachable_cell_dtp(lbs, ubs, ϕ_min, ϕ_max; num_steps = 20)
	cosdmin, cosdmax = cosdextrema(ϕ_min, ϕ_max)
	
	next_lb_x, next_lb_θ, next_lb_y = sim_steps(lbs[1], lbs[2], lbs[3], ϕ_min, cosdmin, num_steps = num_steps)
	next_ub_x, next_ub_θ, next_ub_y = sim_steps(ubs[1], ubs[2], ubs[3], ϕ_max, cosdmax, num_steps = num_steps)
	return [next_lb_x, next_lb_θ, next_lb_y], [next_ub_x, next_ub_θ, next_ub_y]
end