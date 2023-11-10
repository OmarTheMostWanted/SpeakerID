from denoise import AudioDeNoise

audioDenoiser = AudioDeNoise(inputFile="input.wav")
audioDenoiser.deNoise(outputFile="input_denoised.wav")
audioDenoiser.generateNoiseProfile(noiseFile="input_noise_profile.wav")
