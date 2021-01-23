# DCGANs_for_Painting

In order to generate paintings, this program uses generative adversarial networks describe by Goodfellow and others in the following paper : https://arxiv.org/abs/1406.2661.

Here we uses neural networks (MLPs) that we train using generative adversarial training.

The overall model is composed of two sub-models, the generator <img src="https://render.githubusercontent.com/render/math?math=G(\textbf z, \theta_G)"> that takes a latent vector composed of $n$ values from a Gaussian distribution.
And a discriminator <img src="https://render.githubusercontent.com/render/math?math=D(\textbf x, \theta_D)"> that outputs a probability.
G, here, takes a latent vector of lenth 100 and output an vector of length 4096, that can be rearranged in an images of shape 64 * 64.
D, here, takes an image in input and output the probability that the image is in fact a real image or a fake image.
We show alternatively, during training, real images and fake images generated by $G$ to the discriminator. We start by training in the first part of an iteration of training, the discriminator parameters, based on the probability given by D. Next we use the probability given by D to train the parameters of the generator.
It's important to train the discriminator and the generator together, because if the discriminator is too good to recognize real images from fake images at the beginning, the gradients we get for updating the parameters of the generator (computed based on the output of the discriminator) will be too high, leading in an risk of exploding gradients.

The GAN play a minimax game, trying to maximise the probability that D assign the correct label. Et G est entrainé pour minimiser $\log(1-D(\textbf x, \theta_D)$
The GAN objective function is the following : <img src="https://render.githubusercontent.com/render/math?math=\min_G \max_D V(D, G) = E_{x\sim pdata(x)}[logD(x)]+E_{z\sim pz(z)}[log(1\minus D(G(z)))]">
