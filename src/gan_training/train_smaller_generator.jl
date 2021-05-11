include("taxi_models_and_data.jl")
ld = 2
Gconv = BSON.load("conv_generator_results/BCE_BS256_LR0.0007/conv_generator_ld$(ld).bson")[:G]
Gconv = TaxiGConv(Gconv.g_labels |> gpu, Gconv.g_latent |>gpu, Gconv.g_common |> gpu)

D = BSON.load("conv_generator_results/BCE_BS256_LR0.0007/conv_discriminator_ld$(ld).bson")[:D]
D = TaxiDConvSpectral(D.d_labels |> gpu, D.d_common |> gpu)

## Parameters
batch_size = 256
iter = 50000
verbose_freq = 1000

for λ in [0f0, 1f-4, 1f-3, 1f-2, 1f-1, 1f0] # Parameter for the discriminator output
	for lr in [7f-3, 7f-4] # Learning rate
		s = Settings(latent_dim=ld, nclasses=2, G=nothing,D=nothing,loss=nothing, img_fun=nothing, rand_input=nothing, batch_size=batch_size)
		m = TaxiGMLP(Chain(Dense(s.latent_dim + s.nclasses, 256, relu, init=Flux.orthogonal), Dense(256, 256, relu, init=Flux.orthogonal), Dense(256, 256, relu, init=Flux.orthogonal), Dense(256, 256, relu, init=Flux.orthogonal), Dense(256, 16*8, init=Flux.orthogonal), x -> reshape(x, 16,8,1,:)) |> gpu)
		# m = TaxiGMLP(Chain(Dense(s.latent_dim + s.nclasses, 128, relu, init=Flux.orthogonal), Dense(128, 128, relu, init=Flux.orthogonal), Dense(128, 16*8, init=Flux.orthogonal), x -> reshape(x, 16,8,1,:)) |> gpu) # Smaller mlp
		output_dir = "mlp_generator_results/MLP128x2_mae_adv_ld$(ld)_λ$(λ)_LR$(lr)"

		
		data, fixed_noise, fixed_labels = gen_Taxi_images(s)
		θ = Flux.params(m)
		opt = ADAM(lr)
		losses = []
		for i=0:iter
			x, y = Taxi_input(s)
			l, back = Flux.pullback(()-> begin
				xtilde = m(x,y)
				Flux.mae(Gconv(x,y), xtilde) - λ*mean(tanh.(D(xtilde, y))) + orthogonal_regularization(m)
			end, θ)
			if i % verbose_freq == 0
				name = @sprintf("mlpgan_iter_%06d.png", i)
				save(string(output_dir, "/", name), to_image(m, fixed_noise, fixed_labels, s))
				println("iter: $i, loss: ", l)
			end
			push!(losses, l)
			update!(opt, θ, back(1f0))
		end 
		G = TaxiGMLP(m.net |> cpu)
		BSON.@save "$(output_dir)/mlp_generator.bson" G
		BSON.@save "$(output_dir)/losses.bson" losses
	end
end


