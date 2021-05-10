### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 3f7b920e-6c8c-11eb-1bc0-e55f9b4e855f
using PlutoUI

# ╔═╡ 5c755c90-6c8d-11eb-118d-694de45f21c4
using Plots

# ╔═╡ 68d0b3ec-6c8d-11eb-11ed-c7698eadc0c8
md"""
# Dubin's Car Model
"""

# ╔═╡ 75a9037e-6c8d-11eb-0656-0d10f277de0d
md"""
Parameters
"""

# ╔═╡ 7ca2c956-6c8d-11eb-28e0-c58a170fd931
begin
	L = 5 # meters
	v = 5 # m/s
	nothing
end

# ╔═╡ e4ca4704-6c8d-11eb-349f-799dd064b7a0
md"""
Model
"""

# ╔═╡ f6ba63fe-6c8d-11eb-1d4e-67fb69c8ffe4
function next_state(s, ϕ; dt = 0.05)
	# s = [x, θ] = [cte, he]
	# x is in meters and θ is in degrees
	# ϕ is steering angle input (deg)
	x, θ = s
	
	# Dynamics model
	ẋ = v * sind(θ)
	θ̇ = (v / L) * tand(ϕ)
	
	x′ = x + ẋ * dt
	θ′ = θ + θ̇ * dt
	
	return [x′, θ′]
end

# ╔═╡ 0de5c448-6c8f-11eb-2875-e1a10740052b
function next_state(s, y, ϕ; dt = 0.05)
	# for plotting purposes, we might also want the y value
	# s = [x, θ] = [cte, he]
	# x is in meters and θ is in degrees
	# ϕ is steering angle input (deg)
	x, θ = s
	
	# Dynamics model
	ẋ = v * sind(θ)
	ẏ = v * cosd(θ)
	θ̇ = (v / L) * tand(ϕ)
	
	x′ = x + ẋ * dt
	y′ = y + ẏ * dt
	θ′ = θ + rad2deg(θ̇) * dt
	
	return [x′, θ′], y′
end

# ╔═╡ 8e0ccc94-6c8f-11eb-242c-3de6730d1c80
md"""
Simulation
"""

# ╔═╡ 985f921e-6c8f-11eb-334c-8f05ba384d18
function sim_dubins(s₀, y₀, get_steering; num_steps = 500, dt = 0.05, ctrl_every = 20)
	xs = zeros(num_steps + 1)
	θs = zeros(num_steps + 1)
	ys = zeros(num_steps + 1)
	
	xs[1] = s₀[1]
	θs[1] = s₀[2]
	ys[1] = y₀
	
	ϕ = clamp(get_steering(s₀), -60, 60)
	
	for i = 2:num_steps + 1
		s = [xs[i - 1], θs[i - 1]]
		if mod(i - 1, ctrl_every) == 0
			ϕ = clamp(get_steering(s), -60, 60)
		end
		s′, y′ = next_state(s, ys[i - 1], ϕ, dt = dt)
		xs[i] = s′[1]
		θs[i] = s′[2]
		ys[i] = y′
	end
	
	return xs, θs, ys
end	

# ╔═╡ 98ef41dc-6c91-11eb-3a9b-0f44554216f6
md"""
# Plotting
"""

# ╔═╡ a1f56856-6c91-11eb-25ad-6f24fe6a9dbf
function plot_sim(xs, ys)
	p = plot(ys, xs, xlabel = "Downtrack Position", ylabel = "Crosstrack Error", label = "", ylims = (-10, 10))
	plot!(p, ys, zeros(length(xs)), linestyle = :dash, label = "")
	return p
end

# ╔═╡ ad7957b0-6c90-11eb-351c-7d09357eb0c2
md"""
# Control
"""

# ╔═╡ d3d80a00-6c90-11eb-1e5f-21af2406beb8
md"""
k1: $(@bind k1 NumberField(-3:0.01:0, default=0))
k2: $(@bind k2 NumberField(-3:0.01:0, default=0))
"""

# ╔═╡ 7e609550-6c91-11eb-37e9-bb7f6a4b235e
function get_control(s)
	return k1 * s[1] + k2 * s[2]
end

# ╔═╡ e124b84a-6c91-11eb-23fb-6bbe1d53606e
md"""
Initial CTE: $(@bind init_cte NumberField(-10:10, default=0))
"""

# ╔═╡ 93b4b832-6c91-11eb-1dbd-7b7d01523a53
xs, θs, ys = sim_dubins([init_cte, 0.0], 0.0, get_control, ctrl_every = 20, 
	num_steps = 1000);

# ╔═╡ 19245a0e-6c92-11eb-32f3-4bc2293c6e3c
plot_sim(xs, ys)

# ╔═╡ Cell order:
# ╠═3f7b920e-6c8c-11eb-1bc0-e55f9b4e855f
# ╠═5c755c90-6c8d-11eb-118d-694de45f21c4
# ╟─68d0b3ec-6c8d-11eb-11ed-c7698eadc0c8
# ╟─75a9037e-6c8d-11eb-0656-0d10f277de0d
# ╠═7ca2c956-6c8d-11eb-28e0-c58a170fd931
# ╟─e4ca4704-6c8d-11eb-349f-799dd064b7a0
# ╠═f6ba63fe-6c8d-11eb-1d4e-67fb69c8ffe4
# ╠═0de5c448-6c8f-11eb-2875-e1a10740052b
# ╟─8e0ccc94-6c8f-11eb-242c-3de6730d1c80
# ╠═985f921e-6c8f-11eb-334c-8f05ba384d18
# ╟─98ef41dc-6c91-11eb-3a9b-0f44554216f6
# ╠═a1f56856-6c91-11eb-25ad-6f24fe6a9dbf
# ╟─ad7957b0-6c90-11eb-351c-7d09357eb0c2
# ╟─d3d80a00-6c90-11eb-1e5f-21af2406beb8
# ╠═7e609550-6c91-11eb-37e9-bb7f6a4b235e
# ╟─e124b84a-6c91-11eb-23fb-6bbe1d53606e
# ╠═93b4b832-6c91-11eb-1dbd-7b7d01523a53
# ╠═19245a0e-6c92-11eb-32f3-4bc2293c6e3c
