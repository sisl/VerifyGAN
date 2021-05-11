include("taxi_models_and_data.jl")

ld = 2
batch_size = [128, 256, 1024]
epochs = [500, 750, 1000]
learning_rates = [1f-4, 7f-4, 2f-3]
losses = [DCGANLoss(), LSLoss(), WLossGP()]
losses_name = ["BCE", "LS", "WLoss"]
dirs = []

for (loss, loss_name) in zip(losses, losses_name)
	for (bs, nepoch) in zip(batch_size, epochs)
		for lr in learning_rates
			output_dir = "$(loss_name)_BS$(bs)_LR$(lr)"
			push!(dirs, output_dir)
			println("starting $(output_dir)")
			G, D, Ghist, Dhist = train(Settings(G=TaxiGConv, 
								  D=TaxiDConvSpectral, 
								  epochs=nepoch, 
								  batch_size=bs, 
								  rand_input=Taxi_input, 
								  loss=loss, 
								  img_fun=gen_Taxi_images, 
								  nclasses=2, 
								  latent_dim=ld,
								  verbose_freq=10,
								  optD = ADAM(lr, (0.5, 0.99)),
								  optG = ADAM(lr, (0.5, 0.99)),
								  output_dir = "$(output_dir)",))

			G = TaxiGConv(G.g_labels |> cpu, G.g_latent |> cpu, G.g_common |> cpu)
			BSON.@save "$(output_dir)/conv_generator_ld$(ld).bson" G

			D = TaxiDConvSpectral(D.d_labels |> cpu, D.d_common |> cpu)
			BSON.@save "$(output_dir)/conv_discriminator_ld$(ld).bson" D

			BSON.@save "$(output_dir)/d_hist_ld$(ld).bson" Dhist
			BSON.@save "$(output_dir)/g_hist_ld$(ld).bson" Ghist
			BSON.@save "dirs.bson" dirs
		end
	end
end